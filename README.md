SEREngine
=========

Speech Emotion Recognition engine written in C# for xbox games.

Takes recognized speech samples from Microsft's speech recognition engine as input, and outputs the emotionality of the speech sample.

The emotionality (also called excitation) is determined by taking the standard deviation of the fundamental frequency over time.

Works best with voices, as opposed to tones, because it uses the relationship between multiple harmonics to determine fundamental frequency.

Nothing fancy is done to seperate the primary signal from noise, so the pitch can be affected a lot by ambient background noise.

Also, it detects the pitch of male voices much more accurately than female voices

The code is a total mess. I wrote this for a class in college and never went back and cleaned it up.

I also wrote every thing from scratch, including the FFT implementation, because I was too lazy to search for a library to do that stuff. It surely has bugs because you'll see there are no unit tests.

Have fun :)
