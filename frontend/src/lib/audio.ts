/**
 * Client-side audio transcoding for in-page recordings.
 *
 * Why this module exists
 * ---------------------
 * `MediaRecorder` can only emit containers the browser ships an encoder for:
 * Chrome offers `audio/webm;codecs=opus` and `audio/mp4`, Firefox offers
 * `audio/ogg;codecs=opus`, Safari offers `audio/mp4`. The API accepts
 * `.wav .flac .ogg .mp3 .m4a` -- but deliberately *not* `.webm`, because the
 * server decodes through libsndfile, and libsndfile 1.2.2 has no Matroska/WebM
 * support (`sf.available_formats()` lists `OGG` but no `MATROSKA`; `OGG` exposes
 * an `OPUS` subtype, Matroska exposes nothing).
 *
 * The consequence was that the record feature could not work at all in Chrome:
 * every recording was rejected with `UNSUPPORTED_FORMAT`. Widening the server's
 * allow-list would only have moved the failure downstream to a confusing decode
 * error, so the browser transcodes its own output instead.
 *
 * Decoding happens via `AudioContext.decodeAudioData`, which understands every
 * container MediaRecorder can produce. The result is downmixed to mono and
 * resampled to the model's 16 kHz in an `OfflineAudioContext`, then written as
 * 16-bit PCM WAV. That is the format the model consumes natively, so no
 * server-side container sniffing or resampling is needed for a recording.
 *
 * Side benefit: the upload is ~3x smaller than 48 kHz WebM/Opus.
 */

/** Sample rate the model consumes. Kept in step with `AudioConfig.sample_rate`. */
export const MODEL_SAMPLE_RATE = 16000

/** Target for the WAV this module produces. */
const TARGET_RATE = MODEL_SAMPLE_RATE

interface WebkitWindow {
  webkitAudioContext?: typeof AudioContext
}

/** Resolves the prefixed constructor in older Safari. */
function audioContextCtor(): typeof AudioContext {
  const w = window as unknown as WebkitWindow
  const ctor = window.AudioContext ?? w.webkitAudioContext
  if (!ctor) {
    throw new Error('This browser does not support the Web Audio API, so in-page recording cannot be encoded.')
  }
  return ctor
}

/** Decodes any MediaRecorder container into an `AudioBuffer`. */
async function decodeBlob(blob: Blob): Promise<AudioBuffer> {
  const bytes = await blob.arrayBuffer()
  if (bytes.byteLength === 0) {
    throw new Error('The recording captured no audio data.')
  }
  const Ctor = audioContextCtor()
  // A throwaway context purely for decoding. Closing it releases the decode
  // worker; leaving these open is what makes Chrome hit its context limit after
  // a handful of recordings.
  const ctx = new Ctor()
  try {
    // The arrayBuffer form is required because we need an ArrayBuffer, and the
    // promise form is required because the callback form is deprecated.
    return await ctx.decodeAudioData(bytes)
  } catch {
    throw new Error(
      'The browser could not decode its own recording. Try a different browser, or upload a file instead.',
    )
  } finally {
    void ctx.close().catch(() => undefined)
  }
}

/**
 * Downmixes to mono and resamples to `TARGET_RATE`.
 *
 * Uses `OfflineAudioContext` rather than naive sample-dropping so the resample
 * is band-limited; skipping anti-aliasing here would alias down into the speech
 * band and measurably shift the MFCC features the model was trained on.
 */
async function toMonoAtTargetRate(buffer: AudioBuffer, targetRate: number): Promise<Float32Array> {
  const frames = Math.max(1, Math.round(buffer.duration * targetRate))
  const offline = new OfflineAudioContext(1, frames, targetRate)
  const source = offline.createBufferSource()
  source.buffer = buffer
  source.connect(offline.destination)
  source.start()
  const rendered = await offline.startRendering()
  // Copy out: the rendered buffer is only valid for this task's lifetime.
  return new Float32Array(rendered.getChannelData(0))
}

function writeAscii(view: DataView, offset: number, text: string): void {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i))
  }
}

/**
 * Wraps mono float samples in a 16-bit PCM RIFF/WAVE container.
 *
 * Hand-rolled rather than using an encoder dependency: this is the simplest
 * possible container, and it is what the server decodes fastest.
 */
export function encodeWavPcm16(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const channels = 1
  const bytesPerSample = 2
  const blockAlign = channels * bytesPerSample
  const dataSize = samples.length * blockAlign
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)

  writeAscii(view, 0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true) // file size minus the first 8 bytes
  writeAscii(view, 8, 'WAVE')

  writeAscii(view, 12, 'fmt ')
  view.setUint32(16, 16, true) // PCM fmt chunk is always 16 bytes
  view.setUint16(20, 1, true) // format tag 1 = integer PCM
  view.setUint16(22, channels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * blockAlign, true) // byte rate
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, 8 * bytesPerSample, true)

  writeAscii(view, 36, 'data')
  view.setUint32(40, dataSize, true)

  let offset = 44
  for (let i = 0; i < samples.length; i += 1) {
    const s = samples[i] ?? 0
    // Asymmetric integer ranges: -1 maps to -32768, +1 to +32767. Clamping
    // first matters because a decoded buffer can exceed unity after resampling.
    const clipped = s < -1 ? -1 : s > 1 ? 1 : s
    view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true)
    offset += bytesPerSample
  }

  return buffer
}

export interface TranscodedRecording {
  blob: Blob
  /** Duration in seconds, measured from the decoded samples rather than the clock. */
  durationSec: number
  sampleRate: number
  samples: number
}

/**
 * Decodes a recorded blob and re-encodes it as 16 kHz mono WAV.
 *
 * Returns the true sample count, which is a far more trustworthy duration than
 * the wall-clock elapsed time: MediaRecorder can drop or buffer chunks, so the
 * recorded length and the stopwatch routinely disagree by a few hundred ms.
 */
export async function transcodeRecordingToWav(blob: Blob): Promise<TranscodedRecording> {
  const decoded = await decodeBlob(blob)
  if (decoded.length === 0) {
    throw new Error('The recording captured no audio data.')
  }
  const samples = await toMonoAtTargetRate(decoded, TARGET_RATE)
  const wav = encodeWavPcm16(samples, TARGET_RATE)
  return {
    blob: new Blob([wav], { type: 'audio/wav' }),
    durationSec: samples.length / TARGET_RATE,
    sampleRate: TARGET_RATE,
    samples: samples.length,
  }
}