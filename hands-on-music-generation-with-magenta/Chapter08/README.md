# Chapter 8 - Magenta in the browser with Magenta.js

[![Magenta Version 1.1.7](../docs/magenta-v1.1.7-badge.svg)](https://github.com/magenta/magenta/releases/tag/1.1.7)

This chapter will show a JavaScript implementation of Magenta that gained popularity for its ease of use, since it runs in the browser and can be shared as a web page. We'll introduce TensorFlow.js, the technology Magenta.js is built upon, and show what models are available in Magenta.js, including how to convert our previously trained models. Then, we'll create small web applications using GANSynth and MusicVAE for sampling audio and sequences respectively. Finally, we'll see how Magenta.js can interact with other applications, using the Web MIDI API and Node.js.

## Magenta Versioning

```diff
- A newer version of this code is available.
```

This branch shows the code for Magenta v1.1.7, which corresponds to the code in the book. For a more recent version, use the updated [Magenta v2.0.1 branch](https://github.com/PacktPublishing/hands-on-music-generation-with-magenta/tree/magenta-v2.0.1/Chapter08).

## Code 

Before you start, follow the [installation instructions for Magenta 1.1.7](https://github.com/PacktPublishing/hands-on-music-generation-with-magenta/tree/master/Chapter01#installing-magenta).

### [Example 1](chapter_08_example_01.html)

An example on how to use a converted model (from a previously trained model) locally.

### [Example 2](chapter_08_example_02.html)

An example of a Magenta.js web page using GANSynth. Press "Sample GANSynth note" to sample a new note using GANSynth and play it immediately. You can layer as many notes as you want, each note will loop each 4 seconds.

### [Example 2 ES6](chapter_08_example_02_es6.html)

[Example 2](#example-2) using ES6 modules.

### [Example 3](chapter_08_example_03.html)

An example of a Magenta.js web page using MusicVAE. Press "Sample MusicVAE trio" to sample a new sequence of a trio of instruments (drum kit, bass, lead) using MusicVAE and play it immediately. The sequence will loop.

### [Example 4](chapter_08_example_04.html)

An example of a Magenta.js web page using GANSynth and MusicVAE. Press "Sample MusicVAE trio" to sample a new sequence of a trio of instruments (drum kit, bass, lead) using MusicVAE and play it immediately. The sequence will loop. Press "Sample GANSynth note for the lead synth" to sample a new note using GANSynth use it in the lead synth.

### [Example 5](chapter_08_example_05.html)

An example of a Magenta.js web page using the Web Workers API. Press "Sample MusicVAE trio" to sample a new sequence of a trio of instruments (drum kit, bass, lead) using MusicVAE in a Web Worker and play it immediately.

### [Example 6](chapter_08_example_06.html)

An example of a Magenta.js web page using the Web MIDI API. Press "Sample MusicVAE trio" to sample a new sequence of a trio of instruments (drum kit, bass, lead) using MusicVAE and send it to the given MIDI output. The sequence will loop.

### [Example 7](chapter_08_example_07.js)

An example of Magenta.js running in Node.js.

```
node chapter_08_example_07.js
```
