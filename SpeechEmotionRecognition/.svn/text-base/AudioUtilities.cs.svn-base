using System;
using System.Speech;
using System.Speech.Recognition;
using System.Speech.AudioFormat;
using System.IO;

namespace SpeechEmotionRecognition
{
    public class AudioUtilities
    {
        int[][] fastBitReversalTable = null;
        const int MaxFastBits = 16;

        public AudioUtilities()
        {}

        public short[] getArrayFromRecognizedAudio(RecognizedAudio inputAudio)
        {
            SpeechAudioFormatInfo speechAudioFormatInfo = inputAudio.Format;

            // Put the audio into an array.
            // Use a 16 bit short because 16 bits is the max sample size.
            MemoryStream audioStream = new MemoryStream();
            inputAudio.WriteToAudioStream(audioStream);
            byte[] byteArray = audioStream.ToArray();

            /* // For Debugging.
            // Print out the byte audio.
            String output = "audioByteArray: ";
            for (int i = 0; i < byteArray.Length; ++i)
                output += byteArray[i] + ".";
            System.Diagnostics.Debug.WriteLine(output);
            */

            // Convert byteArray[] to short[], keeping channels interleaved.
            long numSamplesInAudio = byteArray.Length / speechAudioFormatInfo.BlockAlign * speechAudioFormatInfo.ChannelCount;
            short[] audioArray = new short[numSamplesInAudio];
            for (int i = 0; i < byteArray.Length; i += speechAudioFormatInfo.BlockAlign / speechAudioFormatInfo.ChannelCount)
            {
                if (speechAudioFormatInfo.BitsPerSample == 16)
                {
                    int audioIndex = i / 2;
                    audioArray[audioIndex] = 0;

                    // The ordering of the bytes for each 16-bit sample is Little-Endian!!!
                    audioArray[audioIndex] |= (short)(byteArray[i + 1] << 8);
                    audioArray[audioIndex] |= (short)byteArray[i];
                }
                else // if (speechAudioFormatInfo.BitsPerSample == 8)
                    audioArray[i] = (short)byteArray[i];
            }

            /* // For Debugging.
            // Print out the short audio.
            output = "audioshortArray: ";
            for (int i = 0; i < numSamplesInAudio; ++i)
                output += audioArray[i] + ".";
            System.Diagnostics.Debug.WriteLine(output);
            */

            return audioArray;
        }

        public void computeSpectrum(short[] audioArray, float[] freqOut, SpeechAudioFormatInfo speechAudioFormatInfo)
        {
            System.Diagnostics.Debug.WriteLine("SpeechEmotionRecognitionEngine::computeSpectrum()");
            if (audioArray == null || freqOut == null)
            {
                System.Diagnostics.Debug.WriteLine("audioArray or freqOut is null");
                return;
            }

            // For Debugging.
            // printDebugFormatInfo(speechAudioFormatInfo);

            // Only allow 8 or 16 bit audio.
            if (speechAudioFormatInfo.BitsPerSample != 8 && speechAudioFormatInfo.BitsPerSample != 16)
            {
                System.Diagnostics.Debug.WriteLine("Invalid BitsPerSample");
                return;
            }

            int windowSize = audioArray.Length;
            int numWindows = audioArray.Length / (windowSize * speechAudioFormatInfo.ChannelCount);
            int height = windowSize / 2;
            int half = windowSize / 2;
            int maxSamples = half;

            float[] processed = new float[windowSize];
            for (int i = 0; i < windowSize; ++i)
                processed[i] = 0.0f;

            float[] fftOut = new float[windowSize];
            float[] inputAudio = new float[windowSize];
            int[] estimatedFundamentalFrequencies = new int[numWindows];

            for (int i = 0; i < windowSize; ++i)
                inputAudio[i] = (float)audioArray[i];

            windowFunction(WindowFunction.HANNING, windowSize, inputAudio);

            // Take FFT.
            fft(inputAudio, null, fftOut, null, 1, speechAudioFormatInfo);

            // Compute power.
            for (int i = 0; i < windowSize; ++i)
                inputAudio[i] = (float)(fftOut[i] * fftOut[i]);

            // Tolonen and Karjalainen recommend taking the cube root
            // of the power, instead of the square root

            for (int i = 0; i < windowSize; i++)
                inputAudio[i] = (float)(Math.Pow(inputAudio[i], 1.0f / 3.0f));

            // Take FFT.
            fft(inputAudio, null, fftOut, null, 1, speechAudioFormatInfo);

            for (int i = 0; i < half; i++)
                processed[i] += fftOut[i];

            // Peak Pruning as described by Tolonen and Karjalainen, 2000

            // Clip at zero, copy to temp array
            for (int i = 0; i < maxSamples; ++i)
            {
                if (processed[i] < 0.0)
                    processed[i] = (float)0.0;
                fftOut[i] = processed[i];
            }

            // Subtract a time-doubled signal (linearly interp.) from the original
            // (clipped) signal
            for (int i = 0; i < maxSamples; ++i)
            {
                if ((i % 2) == 0)
                    processed[i] -= fftOut[i / 2];
                else
                    processed[i] -= ((fftOut[i / 2] + fftOut[i / 2 + 1]) / 2);
            }

            // Clip at zero again
            for (int i = 0; i < maxSamples; ++i)
            {
                if (processed[i] < 0.0)
                    processed[i] = (float)0.0;
            }

            // Find new max
            float max = 0;
            for (int i = 1; i < maxSamples; i++)
                if (processed[i] > max)
                    max = processed[i];

            // Reverse and scale
            for (int i = 0; i < maxSamples; ++i)
                inputAudio[i] = processed[i] / (windowSize / 4);
            for (int i = 0; i < maxSamples; ++i)
                processed[maxSamples - 1 - i] = inputAudio[i];

            // Finally, put it into bins in grayscaleOut[], normalized to a 0.0-1.0 scale

            for (int i = 0; i < height; ++i)
            {
                float bin0 = (float)(i) * maxSamples / height;
                float bin1 = (float)(i + 1) * maxSamples / height;

                float binwidth = bin1 - bin0;

                float value = 0.0f;

                if ((int)bin1 == (int)bin0)
                    value = processed[(int)bin0];
                else
                {
                    value += processed[(int)bin0] * ((int)bin0 + 1 - bin0);
                    bin0 = 1 + (int)bin0;
                    while (bin0 < (int)bin1)
                    {
                        value += processed[(int)bin0];
                        bin0 += 1.0f;
                    }
                    value += processed[(int)bin1] * (bin1 - (int)bin1);

                    value /= binwidth;
                }

                // Should we be clipping at max 1.0? 
                // I trial-and-errored for a while, and I don't think the clipping is necessary.
                // if (value > 1.0)
                //     value = 1.0f;
                if (value < 0.0)
                    value = 0.0f;

                freqOut[i] = value;
            }
        }

        public double computeLogEnergy(short[] audioArray, SpeechAudioFormatInfo speechAudioFormatInfo)
        {
            // Compute power by summing the squares of the signal.
            double energy = 0.0;
            for (int i = 0; i < audioArray.Length; ++i)
                energy += (double)audioArray[i] * (double)audioArray[i];

            energy /= audioArray.Length;
            energy = Math.Log(energy);

            // System.Diagnostics.Debug.WriteLine("energy[]: " + energy);
            return energy;
        }

        // Analyzes the audio currently in the buffer and estimates the fundamental frequency.
        public int extractPitch(short[] audioArray, SpeechAudioFormatInfo speechAudioFormatInfo)
        {
            System.Diagnostics.Debug.WriteLine("SpeechEmotionRecognitionEngine::extractPitch()");
            if (audioArray == null)
            {
                System.Diagnostics.Debug.WriteLine("audioArray is null");
                return -1;
            }

            // For Debugging.
            // printDebugFormatInfo(speechAudioFormatInfo);

            // Only allow 8 or 16 bit audio.
            if (speechAudioFormatInfo.BitsPerSample != 8 && speechAudioFormatInfo.BitsPerSample != 16)
            {
                System.Diagnostics.Debug.WriteLine("Invalid BitsPerSample");
                return -1;
            }

            // To detect the pitch, we take a window of the signal, with a length at least twice as long
            // as the longest period that we might detect. If the sampling rate is 44,100 Hz, this 
            // corresponded to a length of 1200 samples. For effecient calculation, I use this ratio to approximate
            // the windowSize. This give a ratio of 36.75, which I rounded down to 36.
            int windowSize = 2048; //  speechAudioFormatInfo.SamplesPerSecond / 36;
            int numWindows = audioArray.Length / (windowSize * speechAudioFormatInfo.ChannelCount);

            double[][] correlationFunctions = new double[numWindows][];
            int[] estimatedFundamentalFrequencies = new int[numWindows];
            for (int windowIndex = 0; windowIndex < numWindows; ++windowIndex)
            {
                // Store the current window in inputAudio so we can work with it.
                short[] inputAudio = new short[windowSize];
                for (int i = 0; i < windowSize; ++i)
                    inputAudio[i] = audioArray[i + windowIndex * windowSize];

                // Calculate the correlation function.
                correlationFunctions[windowIndex] = correlation(inputAudio, speechAudioFormatInfo);

                // Clip all results below 0 to 0.
                for (int i = 0; i < windowSize / 2; ++i)
                {
                    if (correlationFunctions[windowIndex][i] < 0)
                        correlationFunctions[windowIndex][i] = 0;
                }

                // Stretch correlation results by a factor of 2 and subtract from the original signal.
                for (int i = 0; i < windowSize / 2; ++i)
                {
                    int value;
                    if (i % 2 == 0)
                        value = inputAudio[i / 2];
                    else
                        value = (inputAudio[i / 2 + 1] - inputAudio[i / 2]) / 2;
                    correlationFunctions[windowIndex][i] -= value;
                }
                
                // Clip all results below 0 to 0.
                for (int i = 0; i < windowSize / 2; ++i)
                {
                    if (correlationFunctions[windowIndex][i] < 0)
                        correlationFunctions[windowIndex][i] = 0;
                }

                // Finally, estimate fundamental frequency.
                estimatedFundamentalFrequencies[windowIndex] = estimateF0(correlationFunctions[windowIndex], speechAudioFormatInfo);
            }

            // Calculate the average frequency over all the windows.
            // Can this overflow? Nah, probably not.
            long tmp = estimatedFundamentalFrequencies[0];
            for (int windowIndex = 1; windowIndex < numWindows; ++windowIndex)
                tmp += estimatedFundamentalFrequencies[windowIndex];
            int averageEstimatedFrequency = (int)(tmp / numWindows);

            // Could also return useful information like standard deviation, pitch acceleration, rising/falling, etc.

            return averageEstimatedFrequency;
        }

        public double[] correlation(short[] inputAudio, SpeechAudioFormatInfo speechAudioFormatInfo)
        {
            int size = inputAudio.Length / 2;
            // Initialize the correlation function to 0.
            double[] correlationFunction = new double[size];
            for (int i = 0; i < size; ++i)
                correlationFunction[i] = 0;

            for (int shift = 0; shift < size; shift += speechAudioFormatInfo.ChannelCount)
            {
                for (int audioIndex = 0; audioIndex < size; audioIndex += speechAudioFormatInfo.ChannelCount)
                {
                    // Can overflow happen here when setting an int to the result of multiplying 2 bytes? No, no it can't. Yeah that's right...
                    /*
                    double difference = (double)(inputAudio[audioIndex] - inputAudio[audioIndex + shift * speechAudioFormatInfo.ChannelCount]);
                    correlationFunction[i] += (difference * difference);
                    */
                    correlationFunction[shift] += (double)inputAudio[audioIndex] * (double)inputAudio[audioIndex + shift];
                }
                correlationFunction[shift] /= size;
            }

            /* // For debugging.
            // Print the first window's correlation function, just to see what it looks like.
            String correlationFunctionString = "";
            for (int correlationIndex = 0; correlationIndex < size; ++correlationIndex)
                correlationFunctionString += correlationFunction[correlationIndex] + ".";
            System.Diagnostics.Debug.WriteLine("Correlation function: " + correlationFunctionString);
            */

            return correlationFunction;
        }

        // Estimate the fundamental frequency using the correlation function.
        // Look for the first change in sign -- from negative to positive -- in the differentiated correlationFunction to approximate the fundamental frequency.
        public int estimateF0(double[] corr, SpeechAudioFormatInfo speechAudioFormatInfo)
        {
            int fundamentalPeriodSamples = 0;
            int jitter = 0;
            bool wasNegative = false;

            for (int i = 0; i < corr.Length - 1; ++i)
            {
                if (wasNegative)
                {
                    if (corr[i + 1] - corr[i] >= 0)
                    {
                        if (jitter > 3)
                        {
                            i -= jitter;
                            fundamentalPeriodSamples = i;
                            break;
                        }
                        jitter++;
                    }
                }
                else if (corr[i + 1] - corr[i] <= 0)
                {
                    if (jitter > 3)
                    {
                        wasNegative = true;
                        i -= jitter;
                        jitter = 0;
                        continue;
                    }
                    jitter++;
                }
            }

            int estimatedF0 = 0;
            if (fundamentalPeriodSamples > 0)
                estimatedF0 = (int)(1.0 * speechAudioFormatInfo.SamplesPerSecond / fundamentalPeriodSamples);

            // For debugging.
            System.Diagnostics.Debug.WriteLine("Estimated Fundamental Frequency: " + estimatedF0);

            return estimatedF0;
        }

        public enum WindowFunction { BARTLETT, HAMMING, HANNING, BLACKMAN };
        public void windowFunction(WindowFunction whichFunction, int numSamples, float[] input)
        {
            if (whichFunction == WindowFunction.BARTLETT)
            {
                // Bartlett (triangular) window
                for (int i = 0; i < numSamples / 2; ++i)
                {
                    input[i] *= (float)(i / (numSamples / 2));
                    input[i + (numSamples / 2)] *= (float)(1 - (i / (numSamples / 2)));
                }
            }

            if (whichFunction == WindowFunction.HAMMING)
            {
                // Hamming
                for (int i = 0; i < numSamples; ++i)
                    input[i] *= (float)(0.54 - 0.46 * Math.Cos(2 * Math.PI * i / (numSamples - 1)));
            }

            if (whichFunction == WindowFunction.HANNING)
            {
                // Hanning
                for (int i = 0; i < numSamples; ++i)
                    input[i] *= (float)(0.50 - 0.50 * Math.Cos(2 * Math.PI * i / (numSamples - 1)));
            }

            if (whichFunction == WindowFunction.BLACKMAN)
            {
                // Blackman
                for (int i = 0; i < numSamples; ++i)
                    input[i] *= (float)(0.42 - 0.5 * Math.Cos(2 * Math.PI * i / (numSamples - 1)) + 
                                        0.08 * Math.Cos(4 * Math.PI * i / (numSamples - 1)));
            }
        }

        public bool isPowerOfTwo(int value)
        {
            // Neatest little trick ever =)
            if ((value & (value - 1)) == 0)
                return true;
            return false;
        }

        public int numBitsNeeded(int powerOfTwo)
        {
            if (powerOfTwo < 2)
            {
                System.Diagnostics.Debug.WriteLine("Error: FFT called with size " + powerOfTwo);
                return -1;
            }

            for (int i = 0; ; i++)
                if ((powerOfTwo & (1 << i)) > 0)
                    return i;
        }

        // Fast bit-reversal technique based off of Audacity source.
        public int reverseBits(int index, int NumBits)
        {
            int i, rev;
            for (i = rev = 0; i < NumBits; ++i)
            {
                rev = (rev << 1) | (index & 1);
                index >>= 1;
            }
            return rev;
        }

        // Caluculates look-up table for bit reversal.
        public void initFFT()
        {
            if (fastBitReversalTable != null)
                return;

            fastBitReversalTable = new int[MaxFastBits][];
            for (int i = 0; i < MaxFastBits; ++i)
                fastBitReversalTable[i] = new int[1 << MaxFastBits];

            int len = 2;
            for (int b = 1; b <= MaxFastBits; ++b)
            {
                for (int i = 0; i < len; ++i)
                    fastBitReversalTable[b - 1][i] = reverseBits(i, b);
                len <<= 1;
            }
        }

        // Uses pre-calculated look-up table to reverse the bits.
        public int fastReverseBits(int i, int numBits)
        {
           if (numBits <= MaxFastBits)
               return fastBitReversalTable[numBits - 1][i];
           else
               return reverseBits(i, numBits);
        }

        // Based off of Audacity source and numerical recipes.
        // Also, check out this useful website:
        // http://www.codeproject.com/KB/recipes/howtofft.aspx
        public void fft(float[] realIn, float[] imagIn,
                        float[] realOut, float[] imagOut, int sign,
                        SpeechAudioFormatInfo speechAudioFormatInfo)
        {
            int n, mmax, m, j, istep, i;
            double wtemp, wr, wpr, wpi, wi, theta, tempr, tempi;

            int numSamples = realIn.Length;
            int numBits = numBitsNeeded(numSamples);
            int numComplexSamples = numSamples * 2;

            if (!isPowerOfTwo(numSamples))
            {
                System.Diagnostics.Debug.WriteLine(numSamples + " is not a power of two");
                return;
            }

            if (imagOut == null)
                imagOut = new float[numSamples];

            if (sign > 0)
                sign = 1;
            else
                sign = -1;

            if (fastBitReversalTable == null)
                initFFT();

            // Do simultaneous data copy and bit-reversal ordering into interleaved intermediate output...
            float[] data = new float[numComplexSamples];
            for (i = 0; i < numSamples; i++)
            {
                j = reverseBits(i, numBits);
                data[2 * j] = (float)realIn[i];
                data[2 * j + 1] = (imagIn == null) ? 0.0f : imagIn[i];
            }

            // Do the FFT itself...
            // Danielson-Lanzcos routine
            mmax = 2;
            n = numComplexSamples;
            while (n > mmax)
            {
                istep = mmax << 1;
                theta = sign * (2 * Math.PI / mmax);
                wtemp = Math.Sin(0.5 * theta);
                wpr = -2.0 * wtemp * wtemp;
                wpi = Math.Sin(theta);
                wr = 1.0;
                wi = 0.0;

                for (m = 1; m < mmax; m += 2)
                {
                    for (i = m; i <= n; i += istep)
                    {
                        j = i + mmax;
                        tempr = wr * data[j - 1] - wi * data[j];
                        tempi = wr * data[j] + wi * data[j - 1];
                        data[j - 1] = data[i - 1] - (float)tempr;
                        data[j] = data[i] - (float)tempi;
                        data[i - 1] += (float)tempr;
                        data[i] += (float)tempi;
                    }
                    wtemp = wr;
                    wr = wtemp * wpr - wi * wpi + wr;
                    wi = wi * wpr + wtemp * wpi + wi;
                }
                mmax = istep;
            }

            // De-interleave the real/complex data into the outputs.
            for (i = 0; i < numSamples; ++i)
            {
                realOut[i] = data[2 * i];
                imagOut[i] = data[2 * i + 1];
            }
            
            // Happy =)
        }

        public double getFundamentalFrequency(float[] fftRealOutput, float[] fftComplexOutput, SpeechAudioFormatInfo speechAudioFormatInfo)
        {
            if (fftRealOutput == null ||
                fftComplexOutput == null ||
                speechAudioFormatInfo == null)
                return -1;  // Error.

            int numSamples = fftRealOutput.Length;
            if (fftComplexOutput.Length != numSamples)
                return -1;   // Error.

            // Calculate fundamental frequency.
            int fundamentalFrequencySamples = 0;
            double maxValue = Math.Pow(fftRealOutput[fundamentalFrequencySamples], 2) + Math.Pow(fftComplexOutput[fundamentalFrequencySamples], 2);
            for (int i = 1; i < numSamples; ++i)
            {
                if (Math.Pow(fftRealOutput[i], 2) + Math.Pow(fftComplexOutput[i], 2) > maxValue)
                {
                    fundamentalFrequencySamples = i;
                    maxValue = Math.Pow(fftRealOutput[fundamentalFrequencySamples], 2) + Math.Pow(fftComplexOutput[fundamentalFrequencySamples], 2);
                }
            }
            double fundamentalFrequency = fundamentalFrequencySamples * (speechAudioFormatInfo.SamplesPerSecond / ((double)numSamples / speechAudioFormatInfo.ChannelCount));
            // System.Diagnostics.Debug.WriteLine("fundamentalFrequency: " + fundamentalFrequency);
            
            return fundamentalFrequency;
        }

        public double getMaximumFrequencyValue(float[] fftRealOutput, float[] fftComplexOutput, SpeechAudioFormatInfo speechAudioFormatInfo)
        {
            if (fftRealOutput == null ||
                fftComplexOutput == null)
                return -1;  // Error.

            int numSamples = fftRealOutput.Length;
            if (fftComplexOutput.Length != numSamples)
                return -1;   // Error.

            // Calculate fundamental frequency.
            int fundamentalFrequencySamples = 0;
            double maxValue = Math.Pow(fftRealOutput[fundamentalFrequencySamples], 2) + Math.Pow(fftComplexOutput[fundamentalFrequencySamples], 2);
            for (int i = 1; i < numSamples; ++i)
            {
                if (Math.Pow(fftRealOutput[i], 2) + Math.Pow(fftComplexOutput[i], 2) > maxValue)
                {
                    fundamentalFrequencySamples = i;
                    maxValue = Math.Pow(fftRealOutput[fundamentalFrequencySamples], 2) + Math.Pow(fftComplexOutput[fundamentalFrequencySamples], 2);
                }
            }
            // System.Diagnostics.Debug.WriteLine("maxFrequencyValue: " + maxValue);

            return maxValue;
        }
    }
}
