#chirp parameters (change accordingl to the lua script)
START_FREQ_GHZ      = 77.0
SLOPE_MHZ_PER_US    = 78.9857
IDLE_TIME_US        = 5.0
RAMP_END_TIME_US    = 40.0
ADC_START_TIME_US   = 6.0
ADC_SAMPLES         = 256
SAMPLE_FREQ_KSPS    = 8000.0   # ksps -> 8e6 sps
NCHIRP_LOOPS        = 128      # Nl
START_CHIRP_TX      = 0
END_CHIRP_TX        = 11
NC_CHIRPS_PER_LOOP  = (END_CHIRP_TX - START_CHIRP_TX + 1)


# Speed of light
C0 = 299_792_458.0

# Derived  [UNCHANGED]
FS_FAST = SAMPLE_FREQ_KSPS * 1e3  # Hz
SLOPE_HZ_PER_S = SLOPE_MHZ_PER_US * 1e12  # (MHz/us) -> Hz/s
TC_US = IDLE_TIME_US + RAMP_END_TIME_US
PRF_CHIRP_HZ = 1.0 / (TC_US * 1e-6)
SLOW_FS_LOOPS_HZ = PRF_CHIRP_HZ / NC_CHIRPS_PER_LOOP