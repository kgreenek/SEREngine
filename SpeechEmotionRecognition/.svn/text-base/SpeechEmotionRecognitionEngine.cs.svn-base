using System;
using System.Collections.Generic;
using System.Speech;
using System.Speech.Recognition;
using System.Speech.AudioFormat;
using System.Threading;
using System.ComponentModel;

namespace SpeechEmotionRecognition
{
    public delegate void EmotionRecognizedEvent(int emotion);

    public class EmotionRecognizedEventArgs : EventArgs
    {
        public readonly int emotionLevel;

        public EmotionRecognizedEventArgs(int initEmotionLevel)
        {
            emotionLevel = initEmotionLevel;
        }
    }

    public class SpeechEmotionRecognitionEngine
    {
        List<RecognizedAudio> recognizedAudioQueue = new List<RecognizedAudio>();
        AudioUtilities audioUtilities = new AudioUtilities();
        public EventHandler<EmotionRecognizedEventArgs> EmotionRecognized;
        private readonly SynchronizationContext syncContext;
        Thread thread;

        public SpeechEmotionRecognitionEngine()
        {
            syncContext = AsyncOperationManager.SynchronizationContext;
            thread = new Thread(this.run);
            thread.IsBackground = true;
            thread.Start();
        }

        ~SpeechEmotionRecognitionEngine()
        {
            // Make sure to kill the thread if we started it so it doesn't 
            //   keep the process running when it's trying to exit.
            stop();
        }

        /// <summary>
        /// Called with new audio when a word is detected.
        /// </summary>
        /// <param name="inputAudio"></param>
        public void hereIsAudio(RecognizedAudio inputAudio)
        {
            // Add audio to array to be analyzed.
            lock (recognizedAudioQueue)
            {
                recognizedAudioQueue.Add(inputAudio);
            }
            stateChanged();
        }

        /// <summary>
        /// Scheduler
        /// </summary>
        public bool scheduler()
        {
            lock (recognizedAudioQueue)
            {
                if (recognizedAudioQueue.Count > 0)
                {
                    // Get the audio sample that has been on the queue for the longest.
                    RecognizedAudio recognizedAudio = recognizedAudioQueue[0];
                    recognizedAudioQueue.RemoveAt(0);

                    // Calculate the emotion.
                    int emotion = extractEmotion(recognizedAudio);

                    // Message back with extracted emotion.
                    if (EmotionRecognized != null)
                    {
                        syncContext.Post(delegate { EmotionRecognized(this, new EmotionRecognizedEventArgs(emotion)); }, null);
                    }

                    return true;
                }
            }

            return false;
        }

        public void printDebugFormatInfo(SpeechAudioFormatInfo speechAudioFormatInfoToPrint)
        {
            System.Diagnostics.Debug.WriteLine("Samples per second: " + speechAudioFormatInfoToPrint.SamplesPerSecond);
            System.Diagnostics.Debug.WriteLine("Average bytes per second: " + speechAudioFormatInfoToPrint.AverageBytesPerSecond);
            System.Diagnostics.Debug.WriteLine("Bits per sample: " + speechAudioFormatInfoToPrint.BitsPerSample);
            System.Diagnostics.Debug.WriteLine("Channel count: " + speechAudioFormatInfoToPrint.ChannelCount);
            System.Diagnostics.Debug.WriteLine("Encoding format: " + speechAudioFormatInfoToPrint.EncodingFormat);
            System.Diagnostics.Debug.WriteLine("Block Align: " + speechAudioFormatInfoToPrint.BlockAlign);
        }
        
        /// <summary>
        /// Estimates the emotion present in recognizedAudio
        /// </summary>
        /// <param name="recognizedAudio"></param>
        /// <returns>An int representing the emotion excitement level in the signal</returns>
        public int extractEmotion(RecognizedAudio recognizedAudio)
        {
            System.Diagnostics.Debug.WriteLine("SpeechEmotionRecognitionEngine::extractEmotion()");
            if (recognizedAudio == null)
            {
                System.Diagnostics.Debug.WriteLine("inputAudio is null");
                return -1;
            }

            ///////////////////////////////////////////////////////
            // Extract Features
            ///////////////////////////////////////////////////////
            int windowSize = 2048;
            short[] audioArray = audioUtilities.getArrayFromRecognizedAudio(recognizedAudio);
            int numWindows = audioArray.Length / (windowSize * recognizedAudio.Format.ChannelCount);

            // Calculate duration (in seconds).
            double duration = (double)audioArray.Length / recognizedAudio.Format.SamplesPerSecond;
            // MessageTextBox.AppendText("Audio Duration: " + duration + " seconds\n");

            // Calculate fundamental frequency for each window.
            float[][] freqOut = new float[numWindows][];
            for (int i = 0; i < numWindows; ++i)
                freqOut[i] = new float[windowSize / 2];

            double[] fundamentalFrequencies = new double[numWindows];
            for (int windowIndex = 0; windowIndex < numWindows; ++windowIndex)
            {
                short[] inputAudio = new short[windowSize];
                for (int i = 0; i < windowSize; ++i)
                    inputAudio[i] = audioArray[windowIndex * windowSize + i];

                audioUtilities.computeSpectrum(inputAudio, freqOut[windowIndex], recognizedAudio.Format);
                // for (int i = 0; i < windowSize / 2 - 1; ++i)
                //     System.Diagnostics.Debug.WriteLine("freqOut[" + i + "]: " + freqOut[windowIndex][i]);

                int argmax = 0;
                for (int i = 1; i < windowSize / 2; ++i)
                {
                    if (freqOut[windowIndex][i] > freqOut[windowIndex][argmax])
                        argmax = i;
                }

                double lag = (windowSize / 2 - 1) - argmax;
                fundamentalFrequencies[windowIndex] = recognizedAudio.Format.SamplesPerSecond / lag;
            }
            System.Diagnostics.Debug.WriteLine("Fundamental Frequency: " + fundamentalFrequencies[0]); // * (recognizedAudio.Format.SamplesPerSecond / ((double)(windowSize / 2) / recognizedAudio.Format.ChannelCount)));
            // graphSpectrum(freqOut[0], recognizedAudio.Format);    // This isn't useful.

            // Calculate frequency response for each window.
            float[][] fftRealOutput = new float[numWindows][];
            float[][] fftComplexOutput = new float[numWindows][];
            for (int i = 0; i < numWindows; ++i)
            {
                fftRealOutput[i] = new float[windowSize];
                fftComplexOutput[i] = new float[windowSize];
            }

            for (int windowIndex = 0; windowIndex < numWindows; ++windowIndex)
            {
                // Get float array for current window.
                float[] inputAudio = new float[windowSize];
                for (int i = 0; i < windowSize; ++i)
                    inputAudio[i] = (float)audioArray[windowIndex * windowSize + i];

                // Calculate fft for current window.
                audioUtilities.fft(inputAudio, null, fftRealOutput[windowIndex], fftComplexOutput[windowIndex], 1, recognizedAudio.Format);
                if (fftRealOutput[windowIndex] == null)
                    break;
            }
            // graphFrequencyResponse(fftRealOutput[0], fftComplexOutput[0], recognizedAudio.Format);
            // spectrographForm.drawSpectrograph(fftRealOutput, fftComplexOutput,
            //                                   audioUtilities, recognizedAudio.Format);

            // Calculate the pitch mean.
            double pitchMean = 0;
            int n = 0;
            // MessageTextBox.AppendText("Estimated Fundamental Frequencies: \n");
            for (int i = 0; i < numWindows; ++i)
            {
                // MessageTextBox.AppendText("  " + fundamentalFrequencies[i] + " Hz\n");
                // Only include the fundamental frequencies that are within a reasonable range for human voice.
                if (fundamentalFrequencies[i] > 40 && fundamentalFrequencies[i] < 700)
                {
                    pitchMean += fundamentalFrequencies[i];
                    n++;
                }
            }
            pitchMean /= n;
            // MessageTextBox.AppendText("Pitch Mean: " + pitchMean + "\n");
            // TestPitchExtractionLabel.Text = "Estimated F0: " + pitchMean + " Hz";

            // Calculate pitch standard deviation.
            double pitchStdDev = 0;
            n = 0;
            for (int i = 0; i < numWindows; ++i)
            {
                // Only include the fundamental frequencies that are within a reasonable range for human voice.
                if (fundamentalFrequencies[i] > 40 && fundamentalFrequencies[i] < 700)
                {
                    pitchStdDev += Math.Pow(fundamentalFrequencies[i] - pitchMean, 2);
                    n++;
                }
            }
            pitchStdDev /= n;
            pitchStdDev = Math.Pow(pitchStdDev, 0.5);
            // MessageTextBox.AppendText("Pitch Std Dev: " + pitchStdDev + " Hz\n");

            // Calculate pitch velocity for each window.
            double[] pitchVelocities = new double[numWindows - 1];
            n = 0;
            for (int i = 1; i < fundamentalFrequencies.Length; ++i)
            {
                // Only include the fundamental frequencies that are within a reasonable range for human voice.
                if (fundamentalFrequencies[i] > 40 && fundamentalFrequencies[i] < 700)
                {
                    pitchVelocities[i - 1] = fundamentalFrequencies[i] - fundamentalFrequencies[i - 1];
                    n++;
                }
            }

            // Calculate pitch acceleration for each window.
            double[] pitchAccelerations = new double[numWindows - 1];
            n = 0;
            for (int i = 1; i < pitchVelocities.Length; ++i)
            {
                // Only include the fundamental frequencies that are within a reasonable range for human voice.
                if (fundamentalFrequencies[i] > 40 && fundamentalFrequencies[i] < 700)
                {
                    pitchAccelerations[i - 1] = pitchVelocities[i] - pitchVelocities[i - 1];
                    n++;
                }
            }

            // Calculate average pitch acceleration.
            double averagePitchAcceleration = 0;
            n = 0;
            for (int i = 0; i < pitchAccelerations.Length; ++i)
            {
                // Only include the fundamental frequencies that are within a reasonable range for human voice.
                if (fundamentalFrequencies[i] > 40 && fundamentalFrequencies[i] < 700)
                {
                    averagePitchAcceleration += pitchAccelerations[i];
                    n++;
                }
            }

            averagePitchAcceleration /= n;
            // MessageTextBox.AppendText("Pitch Acceleration: " + averagePitchAcceleration + "\n");

            // Calculate log energy for each window.
            double[] logEnergies = new double[numWindows];
            n = 0;
            for (int windowIndex = 0; windowIndex < numWindows; ++windowIndex)
            {
                short[] inputAudio = new short[windowSize];
                for (int i = 0; i < windowSize; ++i)
                    inputAudio[i] = audioArray[windowIndex * windowSize + i];

                // Only include the fundamental frequencies that are within a reasonable range for human voice.
                if (fundamentalFrequencies[windowIndex] > 40 && fundamentalFrequencies[windowIndex] < 700)
                {
                    logEnergies[windowIndex] = audioUtilities.computeLogEnergy(inputAudio, recognizedAudio.Format);
                    n++;
                }
                // System.Diagnostics.Debug.WriteLine("energy[" + windowIndex + "]: " + logEnergies[windowIndex]);
            }

            // Calculate average log energy.
            double logEnergyMean = 0;
            for (int i = 0; i < numWindows; ++i)
                logEnergyMean += logEnergies[i];
            logEnergyMean /= n;
            // MessageTextBox.AppendText("Log Energy Mean: " + logEnergyMean + "\n");

            // Calculate "Emotion Level" and update GUI.
            double emotionLevel = pitchStdDev * pitchStdDev;

            /*
            EmotionLevelLabel.Text = "Emotion Level: " + emotionLevel + "\n";
            if (emotionLevel <= EmotionLevelProgressBar.Maximum && emotionLevel >= 0)
                EmotionLevelProgressBar.Value = (int)emotionLevel;
            else
                EmotionLevelProgressBar.Value = EmotionLevelProgressBar.Maximum;
            */

            // stateChanged();
            return (int)emotionLevel;
        }

        // Thread specific stuff.
        Semaphore stateChangedSemaphore = new Semaphore(1, 100);    // Set maximum count to be really high. We have no need for a max count. Why does C# impose this argument?...
        bool goOn = true;

        void stateChanged()
        {
            stateChangedSemaphore.Release();
        }

        void run()
        {
            goOn = true;

            while (goOn)
            {
                try
                {
                    // Acquire the semaphore.
                    stateChangedSemaphore.WaitOne();
                    while (scheduler()) ;
                }
                catch
                {
                    System.Diagnostics.Debug.WriteLine("Unexpected exception SpeechEmotionRecognitionEngine\n");
                }
            }
        }

        public void stop()
        {
            goOn = false;
        }
    }
}
