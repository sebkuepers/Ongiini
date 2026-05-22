# Test-fixture provenance

## `suid_afrika.ogg`

A 1.5-second Afrikaans pronunciation of "Suid-Afrika" (Afrikaans for
"South Africa"). Used by `audio_smoke.py` case C to verify the
faster-whisper pipeline transcribes real Afrikaans speech (not just
espeak-ng's robotic synthesis).

- **Source:** [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:%22Suid-Afrika%22,_pronounced_by_Jan_Schutte.oga)
- **Original recording:** Jan Schutte, SABC Transcription Service, 1960
- **License:** Public domain (work of the South African Government,
  copyright expired)
- **Original format:** 24-bit stereo FLAC, 44.1 kHz
- **Our copy:** transcoded with `ffmpeg -ac 1 -ar 16000 -c:a libopus -b:a 32k`
  to mono OGG/Opus 16 kHz @ 32 kbps — matches WhatsApp's voice-note
  container/codec shape.
