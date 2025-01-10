import unittest
from miscfunc_40 import next_n_days, next_day
import pandas as pd

class TestNextNDays(unittest.TestCase):
    
    def test_normal_case(self):
        """Test normal case with valid input"""
        result = next_n_days("2024-01-01", 3)
        expected = pd.date_range(start="2024-01-02", periods=3, freq='D')
        self.assertTrue((result == expected).all())
        
    def test_negative_days(self):
        """Test with negative days input"""
        with self.assertRaises(ValueError):
            next_n_days("2024-01-01", -1)
                     
    def test_invalid_date_format(self):
        """Test with invalid date format"""
        with self.assertRaises(ValueError):
            next_n_days("01-01-2024", 3)
            
    def test_non_integer_days(self):
        """Test with non-integer days input"""
        with self.assertRaises(TypeError):
            next_n_days("2024-01-01", 3.5)

class TestNextDay(unittest.TestCase):
    
    def test_normal_case(self):
        """Test normal case with valid input"""
        result = next_day("2024-01-01")
        expected = pd.Timestamp("2024-01-02")
        self.assertEqual(result, expected)
        
    def test_end_of_month(self):
        """Test end of month case"""
        result = next_day("2024-01-31")
        expected = pd.Timestamp("2024-02-01")
        self.assertEqual(result, expected)
        
    def test_end_of_year(self):
        """Test end of year case"""
        result = next_day("2024-12-31")
        expected = pd.Timestamp("2025-01-01")
        self.assertEqual(result, expected)
        
    def test_invalid_date_format(self):
        """Test with invalid date format"""
        with self.assertRaises(ValueError):
            next_day("01-01-2024")

import numpy as np
import matplotlib.pyplot as plt
from miscfunc_40 import powersp

class TestPowerSpectrum(unittest.TestCase):
    
    def test_sine_wave(self):
        """Test power spectrum of a simple sine wave"""
        # Generate test signal
        fs = 1000  # Sampling frequency
        t = np.linspace(0, 1.0, fs, endpoint=False)
        freq = 50.0  # Signal frequency
        signal = np.sin(2 * np.pi * freq * t)
        
        # Compute power spectrum
        freq_axis, power = powersp(signal)
        
        # Verify peak frequency
        peak_idx = np.argmax(power)
        self.assertAlmostEqual(freq_axis[peak_idx], freq, delta=1.0)
        
        # Plot results for visual verification
        plt.figure()
        plt.plot(freq_axis, power)
        plt.title('Power Spectrum of 50 Hz Sine Wave')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power')
        plt.grid(True)
        plt.savefig('power_spectrum_test.png')
        plt.close()

    def test_white_noise(self):
        """Test power spectrum of white noise"""
        # Generate test signal
        fs = 1000
        t = np.linspace(0, 1.0, fs)
        signal = np.random.normal(0, 0.1, fs)
        
        # Compute power spectrum
        freq_axis, power = powersp(signal)
        
        # Verify flat spectrum
        mean_power = np.mean(power)
        std_power = np.std(power)
        self.assertLess(std_power / mean_power, 1.2)  # Check flatness

if __name__ == '__main__':
    unittest.main()
