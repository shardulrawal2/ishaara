<div align="center">

# इशारा · Ishaara

### From signs to shared understanding.

**A low-cost, hands-free communication bridge designed for Indian Sign Language users.**

[![Status](https://img.shields.io/badge/status-prototype_in_development-f59e0b?style=for-the-badge)](#project-status)
[![Focus](https://img.shields.io/badge/focus-Indian_Sign_Language-7c3aed?style=for-the-badge)](#why-ishaara)
[![Prototype Cost](https://img.shields.io/badge/prototype_cost-₹1.2K–₹1.8K-16a34a?style=for-the-badge)](#hardware)

</div>

---

## The communication gap

India's Deaf and Hard-of-Hearing community numbers in the tens of millions, while the country has only a few hundred certified Indian Sign Language (ISL) interpreters. An interpreter-dependent solution cannot realistically be present for every conversation, classroom, clinic, shop, or emergency.

Most technology addresses only half of the exchange: it translates signs into text or speech, but gives the Deaf user no equally natural way to receive the hearing person's reply.

**Ishaara is being designed to close that gap.**

It combines a clip-on camera module with a companion phone app to translate signed gestures into speech. Its longer-term architecture adds speech-to-caption translation, creating a genuinely two-way communication loop.

> Ishaara does not claim to invent sign recognition. It focuses on making assistive communication affordable, wearable, hands-free, and purpose-built for ISL.

## How it works

```mermaid
flowchart LR
    A[Deaf user signs] --> B[Clip-on camera]
    B -->|Wi-Fi video stream| C[Companion app]
    C --> D[Pose landmarks]
    D --> E[Sequence classifier]
    E --> F[Recognized text]
    F --> G[Text-to-speech]
    G --> H[Hearing listener]

    H -. Future: spoken reply .-> I[Speech-to-text]
    I -.-> J[Captions for Deaf user]
```

The wearable stays deliberately lightweight: it captures and streams video while the phone handles pose extraction, inference, language processing, and text-to-speech. This keeps the hardware affordable and makes future model updates easier to deliver.

## Why Ishaara?

| Design goal | Ishaara's approach |
|---|---|
| **Affordable** | Target prototype hardware cost of approximately **₹1,200–₹1,800** |
| **Hands-free** | A small module clips onto existing glasses instead of requiring a phone to be held up |
| **ISL-first** | Designed around Indian Sign Language rather than treating ASL or BSL as interchangeable |
| **Two-way by design** | Sign-to-speech is the MVP; speech-to-caption is the planned return path |
| **Resilient** | Emergency phrases can work without depending on successful ML recognition |
| **Personal** | Custom gesture training can support names, local expressions, and individual signing styles |

## MVP scope

The first prototype intentionally targets a **small, well-tested vocabulary**, not unrestricted conversational ISL.

- **Sign → text → speech:** recognizes a scoped set of approximately 50–200 trained signs
- **Fingerspelling mode:** supports words outside the trained vocabulary, letter by letter
- **Emergency gesture:** triggers a preloaded phrase such as “I need help” without relying on the full recognition pipeline
- **Custom gestures:** lets a user record personal signs through the companion app
- **Recognition feedback:** confirms successful recognition through visual or haptic feedback
- **Companion app:** manages streaming, inference, mode selection, training, and speech output

## Software pipeline

1. **Capture** — the ESP32-CAM captures JPEG frames and streams them to the phone over Wi-Fi.
2. **Pose extraction** — MediaPipe Holistic converts pixels into hand, face, and body landmarks.
3. **Sequence buffering** — the app collects roughly 1–2 seconds of landmarks because a sign is a movement, not a single pose.
4. **Classification** — an LSTM or transformer-based model predicts a trained sign and confidence score.
5. **Text assembly** — recognized signs map to words or phrases.
6. **Speech synthesis** — native mobile text-to-speech produces audio.
7. **Delivery** — the phone speaker or a Bluetooth earpiece plays the translated phrase.

### Expected latency

The target end-to-end delay is approximately **1.5–2.5 seconds in favourable conditions**, with a possible **5–6 second delay** on weaker networks or budget phones. The interface should communicate this state clearly rather than making the user wonder whether recognition failed.

## Hardware

| Component | Purpose | Approx. cost |
|---|---|---:|
| ESP32-CAM (OV2640) | Image capture and wireless streaming | ₹350–₹450 |
| FTDI USB-to-TTL programmer | Firmware flashing; reusable | ₹80–₹200 |
| TP4056 charging module | Safe Li-ion charging | ₹11–₹15 |
| 3.7 V, 500 mAh LiPo battery | Portable power | ₹150–₹250 |
| Micro toggle switch | Physical power control | ₹25 |
| 3D-printed glasses clip | Attachment to existing eyewear | ₹100–₹300 |
| Bluetooth earpiece | Private audio output | ₹300–₹600 |

**Estimated prototype total: ₹1,200–₹1,800**  
Using the phone speaker can reduce a basic build to roughly **₹700–₹1,000**.

## Project status

> [!IMPORTANT]
> Ishaara is currently a prototype concept in active development. The capabilities below are targets, not claims of production readiness or clinical-grade reliability.

| Stage | Goal | Status |
|---|---|---|
| 1 | ESP32-CAM video stream to phone | Planned |
| 2 | Landmark extraction and scoped sign classifier | Planned |
| 3 | Text-to-speech and emergency phrases | Planned |
| 4 | Companion app and custom gesture flow | Planned |
| 5 | Speech-to-caption return path | Roadmap |
| 6 | Continuous ISL and group conversations | Research roadmap |

## 24-hour prototype plan

| Time | Focus |
|---|---|
| Hours 0–6 | Flash hardware and establish stable camera streaming |
| Hours 6–14 | Integrate MediaPipe and train a scoped gesture classifier |
| Hours 14–20 | Connect the app, fingerspelling, emergency phrase, and custom gesture flows |
| Hours 20–24 | Add TTS, test end-to-end latency, and polish the demo |

## Roadmap

- **Speech → captions:** display a hearing person's reply for the Deaf wearer
- **Sound awareness:** detect alarms, doorbells, and name-calling, then provide haptic alerts
- **Expanded ISL vocabulary:** move beyond the small MVP set as representative data becomes available
- **Continuous signing:** progress from isolated signs toward sentence-level recognition
- **Improved hardware:** evaluate higher-frame-rate camera platforms such as the ESP32-P4
- **Multi-person conversations:** distinguish speakers and signers in classrooms, meetings, and groups

## Known limitations

We believe responsible assistive technology starts with honest boundaries.

- Continuous natural sign-language recognition remains an open research problem; the MVP will recognize only a scoped vocabulary.
- The ESP32-CAM is limited to roughly 10 fps at QVGA in this use case, which constrains fast and detailed signing.
- Accuracy will vary with lighting, framing, distance, signer consistency, and the quality and representativeness of training data.
- A 1–2 second sequence buffer is inherent to motion-based recognition, so the experience will be assistive rather than instant.
- ISL-specific datasets are limited. Any non-ISL proxy data used during early experimentation must be disclosed and replaced through ethical, community-informed data collection.
- A hearing collaborator, interpreter, or ISL-fluent tester must remain part of validation; technical metrics alone cannot establish usefulness.

## What makes this worth building

Ishaara aims to make everyday communication possible without requiring a human interpreter to be physically present—using consumer-grade hardware at approximately ₹1,500 instead of assistive devices that can cost lakhs.

The product is meant to serve the Deaf wearer, not merely translate for the hearing person. Recognition feedback, emergency communication, future speech captions, and sound-event alerts are central to that promise.

## Team needs

We are looking for contributors and mentors across:

- **Embedded systems** — ESP32 firmware, streaming, battery, and enclosure design
- **ML / computer vision** — landmark pipelines, temporal classifiers, evaluation, and dataset strategy
- **Mobile engineering** — streaming, on-device inference, TTS, captions, and accessibility
- **Product design** — low-friction interactions for Deaf and hearing participants
- **ISL and Deaf-community expertise** — language accuracy, lived-experience validation, and responsible testing

## Contributing

Ishaara is at an early stage, so focused experiments and candid feedback are especially valuable. Before opening a large pull request, start a discussion or issue describing the user need, proposed approach, and how it will be tested with ISL users.

Please avoid presenting ASL/BSL demonstrations as proof of ISL performance. Contributions involving signer data should include clear consent, privacy, storage, and deletion practices.

---

<div align="center">

### इशारा — because communication should never depend on who is available to interpret.

Built with empathy, tested with honesty, and shaped with the Deaf community.

</div>
