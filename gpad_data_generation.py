import numpy as np
import matplotlib.pyplot as plt
import madmom
import pywt
import pywt.data
from scipy import signal
from pydub import AudioSegment

def convert_mp3_to_wav(filename):
    """
    converts a mp3 file to a wav file
    
    Arguments:
    filename -- string with the name of the mp3 file
                e.g.: filename = "/Music/saudade.mp3"
    
    Returns: 
    /~/filename.wav
    """
    sound = AudioSegment.from_mp3(filename)
    soundwav = sound.export(filename[:-4] + ".wav", format="wav")
    soundwav.close()

def music_loading(filename, sample_rate=44100):
    """
    loads one music to the memory as a normalized numpy array
    
    Arguments:
    filename    -- string with the name of the mp3 file
                   e.g.: filename = "/Music/saudade.mp3"
    sample_rate -- [OPTIONAL] sample rate of the music in Hertz
    
    Returns: 
    musica      -- a numpy array of dimensions (n,1) with n the number of samples of the music
    """
    
    sig = madmom.audio.signal.Signal(filename, sample_rate = sample_rate, num_channels = 1)
    musica = np.array(sig)
    musica = musica/np.max(np.abs(musica))
    return musica
    
def ODF_SuperFlux(signal, sample_rate, frame_size, hop):
    """
    computes the Onset Detection Function of a signal
    
    Arguments:
    signal      -- a numpy array of dimensions (n,1)
    sample_rate -- sample rate of the signal in Hertz
    frame_size  -- size in samples of the frames that will build the spectogram
    hop         -- distance in samples between the frame i+1 and the frame i
    
    Returns: 
    ODF         -- a numpy array of dimension (n,1) with n the number of samples of the ODF
    """
    
    # Loading a file with desired sample_rate and down-mixed
    sig = madmom.audio.signal.Signal(signal, sample_rate = sample_rate, num_channels = 1)

    # We frame the signal
    frames = madmom.audio.signal.FramedSignal(sig, frame_size=frame_size, hop=hop)

    # We compute the log filtered spectrogram of each frame, the number of bands is fixed based in the documentation
    log_filt_spec = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(frames, num_bands = 24)

    # We compute the ODF presented in Bock et tal (2013)
    odf_raw = madmom.features.onsets.superflux(log_filt_spec)
    odf_raw = np.array(odf_raw)
    ODF = odf_raw / np.max(odf_raw)
    
    return ODF

def PeDF(ODF, form):
    """
    computes the Periodicity Function, a normalized autocorrelation of the ODF signal
    
    Arguments:
    ODF         -- a numpy array of dimensions (n,1)
    form        -- definition of PeDF calculation
                    'full'    -> The output will be a symetric signal, the full autocorrelation
                    'partial' -> The output will be one half of the full autocorrelation
    
    Returns: 
    PeDF        -- a numpy array of dimension (n,1) with n the number of samples of the PeDF
    """
    
    corr = signal.correlate(ODF, ODF, mode='same', method='direct')
    corr = corr / np.max(corr)
    
    if form == "full":
        PeDF = corr
        return PeDF
    if form == "partial":
        index = np.where(corr == 1)
        index = index[0][0]
        PeDF = corr[index:]
        return PeDF
    
def preprocess(signal_we_want_to_analyze, nivel_wavelet):
    """
    realizes the preprocessing of a signal to output its ODFs, PeDFs and wavelet coefficients
    
    Arguments:
    signal_we_want_to_analyze  -- a numpy array of dimensions (n,1)
    nivel_wavelet              -- the level of the wavelet transform
    
    Returns: 
    ODF_SET                    -- a list containing ODFs functions
    PeDF_FULL_SET              -- a list containing the whole PeDFs functions
    PeDF_PARTIAL_SET           -- a list containing the halfs of the PeDFs functions
    coeffs                     -- the wavelet coefficients of the wavelet transform decomposition over "signal_we_want_to_analyze"
    
    """
    
    # Wavelet Decomposition
    N = nivel_wavelet
    coif3 = pywt.Wavelet('coif3')
    coeffs = pywt.wavedec(signal_we_want_to_analyze, coif3, level = N) #cAN, cDN, cD(N-1), cD(N-2), .... based in level = N

    # ODF Generation

    # Empty list of ODFs
    ODF_SET = []
    for index in range(0,len(coeffs)):
        coeff_to_odf = coeffs[index]
        sample_rate = 44100
        frame_size = 4096 #92.8ms

        if index == 0:
            samplerate_equivalente = sample_rate/2**(len(coeffs)-1)
            frame_size_equivalente = frame_size/2**(len(coeffs)-1)
        else:
            samplerate_equivalente = sample_rate/2**(len(coeffs)-index)
            frame_size_equivalente = frame_size/2**(len(coeffs)-index)

        ODF = ODF_SuperFlux(coeff_to_odf, samplerate_equivalente, frame_size_equivalente, frame_size_equivalente/2)
        ODF_SET.append(ODF)

    # PeDF Generation 
    PeDF_FULL_SET = []
    PeDF_PARTIAL_SET = []
    for index in range(0,len(ODF_SET)):
        PeDF1 = PeDF(ODF_SET[index], form = "full")
        PeDF2 = PeDF(ODF_SET[index], form = "partial")
        PeDF_FULL_SET.append(PeDF1)
        PeDF_PARTIAL_SET.append(PeDF2)
    
    return ODF_SET, PeDF_FULL_SET, PeDF_PARTIAL_SET, coeffs

def music_processor(filename, start_sample=None, final_sample=None, sample_rate=None, nivel_wavelet=None):
    """
    realizes the preprocessing of a music file's segment to output its ODFs, PeDFs and wavelet coefficients
    
    Arguments:
    filename                   -- string with the name of the mp3 file
                                  e.g.: filename = "/Music/saudade.mp3"
    start_sample               -- [OPTIONAL] starting sample of the audio segment exctracted of "file"
    final_sample               -- [OPTIONAL] final sample of the audio segment exctracted of "file"
    sample_rate                -- [OPTIONAL] the sample rate of "file"
    nivel_wavelet              -- [OPTIONAL] the level of the wavelet transform decomposition
    
    Returns: 
    ODF_SET                    -- a list containing ODFs functions
    PeDF_FULL_SET              -- a list containing the whole PeDFs functions
    PeDF_PARTIAL_SET           -- a list containing the halfs of the PeDFs functions
    coeffs                     -- the wavelet coefficients of the wavelet transform over "signal_we_want_to_analyze"
    
    """
    
    musica_inteira = music_loading(filename, sample_rate)
    musica = musica_inteira[start_sample:final_sample]
    
    ODF_SET, PeDF_FULL_SET, PeDF_PARTIAL_SET, coeffs = preprocess(musica, nivel_wavelet)
    
    return ODF_SET, PeDF_FULL_SET, PeDF_PARTIAL_SET, coeffs

def plot_ODF(ODF, index=None, size=None):
    """
    realizes the plotting of the ODF
    
    Arguments:
    ODF        -- a numpy array of dimensions (n,1)
    index      -- the ODF level corresponding to its wavelet coefficients
    size       -- the level of the wavelet decomposition that generated all the ODFs
    
    Returns: 
    the plot of the ODF in (samples,Onset Stregth) 
    
    """
    
    
    plt.figure()
    plt.plot(ODF)
    plt.title('ODF')
    plt.ylabel('Onset Strength')
    plt.xlabel('Samples')
        
    if index and size != None:
            if index == 0:
                name = "cA" + str(size-1)
                plt.title('ODF - ' + name)
            else:
                name = "cD" + str(size-index)
                plt.title('ODF - ' + name)
    else:
        plt.title('ODF')
    
def plot_PeDF(PeDF, form, index=None, size=None):
    
    """
    realizes the plotting of the PeDF
    
    Arguments:
    ODF        -- a numpy array of dimensions (n,1)
    form       -- 'total' or 'partial', related to the PeDF's aspect
    index      -- the PeDF level corresponding to its wavelet coefficients
    size       -- the level of the wavelet decomposition that generated all the PeDFs
    
    Returns: 
    the plot of the PeDF in (lags,autocorrelation probability) 
    
    """
    
    if form == "full":
        plt.figure()
        x1 = np.arange(-len(PeDF)/2, len(PeDF)/2,1)
        plt.plot(x1,PeDF)
        
        if index and size != None:
            if index == 0:
                name = "cA" + str(size-1)
                plt.title('PeDF - Total - ' + name)
            else:
                name = "cD" + str(size-index)
                plt.title('PeDF - Total - ' + name)
        else:
            plt.title('PeDF - Total')
        plt.xlabel('Lags')
            
    if form == "partial":
        plt.figure()
        plt.plot(PeDF)
        
        if index and size != None:
            if index == 0:
                name = "cA" + str(size-1)
                plt.title('PeDF - Partial - ' + name)
            else:
                name = "cD" + str(size-index)
                plt.title('PeDF - Partial - ' + name)
        else:
            plt.title('PeDF - Total')
        plt.xlabel('Lags')
        
