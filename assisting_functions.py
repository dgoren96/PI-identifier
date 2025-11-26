import time
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from skimage import filters
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter
from skimage.feature import canny
from matplotlib import gridspec
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import os
import random
from scipy.signal import resample
from skimage.morphology import skeletonize, remove_small_objects
import numpy as np
from skimage.filters import gaussian
from scipy.fft import fft, ifft


def update_cursor(event):
    """
    Update the cursor position and on-figure display based on mouse movement.

    This function is triggered by a matplotlib event (e.g., mouse motion).
    When the mouse is inside the plot axes, the function:
        - Rounds and stores the current mouse (x, y) data coordinates.
        - Moves an existing point marker (`cursor_line`) to the new location.
        - Updates a text label (`cursor_text`) showing the current coordinates.
        - Redraws the figure to reflect the updated cursor display.

    Parameters
    ----------
    event : matplotlib.backend_bases.MouseEvent
        The event object containing mouse position and context information.
        Only events occurring inside the axes (`event.inaxes`) are processed.

    Notes
    -----
    This function relies on global variables:
        - x_coord, y_coord : store the current cursor coordinates.
        - cursor_line      : a matplotlib Line2D object used to show the cursor point.
        - cursor_text      : a matplotlib Text object displaying coordinate values.
    """
    global x_coord, y_coord
    if event.inaxes:
        x_coord, y_coord = round(event.xdata), round(event.ydata)
        cursor_line.set_data([x_coord], [y_coord])
        cursor_text.set_text(f'x: {x_coord:.0f}, y: {y_coord:.0f}')
        plt.draw()

# Function to map resized pixel coordinates to original image coordinates
def get_original_coordinates(resized_x, resized_y, width_ratio, height_ratio):
    """
    Map coordinates from a resized image back to the original image.

    Parameters
    ----------
    resized_x : int or float
        X-coordinate in the resized image.
    resized_y : int or float
        Y-coordinate in the resized image.
    width_ratio : float
        Ratio of resized width to original width (resized_width / original_width).
    height_ratio : float
        Ratio of resized height to original height (resized_height / original_height).

    Returns
    -------
    tuple
        (orig_x, orig_y) coordinates in the original image.
    """
    orig_x = int(resized_x / width_ratio)
    orig_y = int(resized_y / height_ratio)
    return orig_x, orig_y


def display_image_with_cross_cursor(image, title):
    """
    Display an image with an interactive crosshair cursor and capture user clicks.

    This function opens an interactive matplotlib window where the user can:
        - Move the mouse to update a red crosshair cursor on the image.
        - Left / Right / Middle click anywhere on the image to select a point.
        - Automatically close the figure upon clicking.

    The function returns the (x, y) coordinates of the last mouse position
    at the moment of clicking, along with the type of click performed.

    Parameters
    ----------
    image : ndarray
        The grayscale image to be displayed.
    title : str
        The title displayed above the figure.

    Returns
    -------
    tuple
        (x_coord, y_coord, event_type)
        x_coord : float
            X coordinate of the click.
        y_coord : float
            Y coordinate of the click.
        event_type : str
            One of: 'left_click', 'right_click', 'middle_click'.

    Notes
    -----
    - Uses global variables:
        cursor_line, cursor_text : updated by `update_cursor()`.
        cursor_position          : tracks current cursor position.
        event_type               : stores the click type.
    - Mouse motion continuously updates the cursor marker using `on_motion()`.
    - A click event closes the figure and finalizes the selected coordinates.
    """

    global cursor_line, cursor_text, event_type, cursor_position
    time.sleep(0.5)
    cursor_position = [0, 0]

    # Create figure and display the image
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.imshow(image, cmap='gray')
    ax.set_title(title)

    # Initialize the crosshair marker and on-screen coordinate text
    cursor_line, = ax.plot([], [], 'r+', markersize=10)
    cursor_text = ax.text(0.5, 0.9, '', color='red',
                          transform=ax.transAxes, ha='center')
    plt.draw()

    def on_click(event):
        """Handle mouse clicks and close the figure."""
        global event_type, cursor_position
        if event.button == 1:  # Left click
            cursor_position = (event.xdata, event.ydata)
            plt.close(fig)
            event_type = 'left_click'
        elif event.button == 3:  # Right click
            cursor_position = (event.xdata, event.ydata)
            plt.close(fig)
            event_type = 'right_click'
        elif event.button == 2:  # Middle click
            cursor_position = (event.xdata, event.ydata)
            plt.close(fig)
            event_type = 'middle_click'

    def on_motion(event):
        """Handle mouse movement and update the crosshair cursor."""
        global cursor_position
        if event.xdata is not None and event.ydata is not None:
            cursor_position = (event.xdata, event.ydata)
            update_cursor(event)

    # Event connections
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.ion()
    plt.tight_layout()
    plt.show(block=True)  # Pause execution until user clicks & closes

    return cursor_position[0], cursor_position[1], event_type


def af_filter(noisy_image, type='wiener', psf_size=(9, 9), mb_ker=15):
    """
    Apply a denoising filter (median, Gaussian, or Wiener) to an input image.

    This function provides three filtering modes:
        - 'median'  : Median filtering using OpenCV.
        - 'gaussian': Gaussian blur using OpenCV.
        - 'wiener'  : Custom Wiener filtering implemented via FFT.

    For the Wiener filter:
        - Noise variance is estimated using a Gaussian-blurred version of the image.
        - A Gaussian point-spread function (PSF) is assumed.
        - Filtering is performed in the frequency domain.

    Parameters
    ----------
    noisy_image : ndarray
        Input noisy image to be filtered.
    type : str, optional
        The filtering method to apply. Options:
        'median', 'gaussian', 'wiener'. Default is 'wiener'.
    psf_size : tuple, optional
        Size of the Gaussian kernel (for Gaussian and Wiener filters).
        Default is (9, 9).
    mb_ker : int, optional
        Kernel size for median filter (OpenCV requirement: odd integer).
        Default is 15.

    Returns
    -------
    w : ndarray
        The filtered image.
    w_size : tuple
        Dimensions of the filtered image (height, width).

    Notes
    -----
    - The custom Wiener filter is normalized to 0–255 and returned as uint8.
    - If an unsupported filter type is given, the function prints an error
      and returns 0 with a size corresponding to that value.
    """

    if type == 'median':
        w = cv2.medianBlur(noisy_image, mb_ker)

    elif type == 'gaussian':
        w = cv2.GaussianBlur(noisy_image, psf_size, 0)

    elif type == 'wiener':
        # Estimate noise variance using Gaussian blur
        noise = noisy_image - cv2.GaussianBlur(noisy_image, psf_size, 0)
        estimated_noise_var = np.var(noise)

        # Fourier transform of the noisy image
        F = np.fft.fft2(noisy_image)

        # Create Gaussian PSF
        psf = cv2.getGaussianKernel(psf_size[0], -1)
        psf_ft = np.fft.fft2(psf, s=noisy_image.shape)

        # Wiener filter in the frequency domain
        H_star = np.conj(psf_ft)
        denominator = np.abs(psf_ft)**2 + estimated_noise_var
        wiener_filtered = F * H_star / denominator

        # Inverse FFT and magnitude
        wiener_filtered_abs = np.abs(np.fft.ifft2(wiener_filtered))

        # Normalize output to 0–255 range
        w = ((wiener_filtered_abs - wiener_filtered_abs.min()) /
             (wiener_filtered_abs.max() - wiener_filtered_abs.min()) * 255).astype(np.uint8)

    else:
        w = 0
        print('ERROR: only median, gaussian and wiener are available')

    w_size = w.shape
    return w, w_size



def plot_all_figures(gray_im, binaryImage, Canny_img, binaryImage_median,
                     sig_out_mean, t_mean, sig_out_canny, t_canny,
                     sig_out_median, t_median):
    """
    Plot a set of image-processing results and their corresponding 1D signals.

    This function displays:
        - Four images (original grayscale crop, mean-filtered binary image,
          Canny edge image, and median-filtered binary image).
        - Three plots showing intensity signals extracted from each filtering method.

    Layout (4×4 grid):
        Row 1: Cropped image | Mean-filtered | Canny | Median-filtered
        Row 2: Mean filter signal (full width)
        Row 3: Canny signal (full width)
        Row 4: Median filter signal (full width)

    Parameters
    ----------
    gray_im : ndarray
        The cropped grayscale input image.
    binaryImage : ndarray
        Binary image after mean/average filtering.
    Canny_img : ndarray
        Image after Canny edge detection.
    binaryImage_median : ndarray
        Binary image obtained after median filtering.
    sig_out_mean : array-like
        1D signal extracted from the mean-filtered result.
    t_mean : array-like
        Time or x-axis corresponding to `sig_out_mean`.
    sig_out_canny : array-like
        Signal extracted from the Canny image.
    t_canny : array-like
        Time or x-axis corresponding to `sig_out_canny`.
    sig_out_median : array-like
        Signal extracted from the median-filtered image.
    t_median : array-like
        Time or x-axis corresponding to `sig_out_median`.

    Notes
    -----
    - Uses `matplotlib.gridspec` for structured layout.
    - All figures are shown together in one window.

    """

    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(4, 4, width_ratios=[1, 1, 1, 1])

    # Plot the images
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(gray_im, cmap='gray')
    ax1.set_title('Cropped Image')

    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(binaryImage, cmap='gray')
    ax2.set_title('Mean filter, pre smoothed')

    ax3 = plt.subplot(gs[0, 2])
    ax3.imshow(Canny_img, cmap='gray')
    ax3.set_title('Canny Image')

    ax4 = plt.subplot(gs[0, 3])
    ax4.imshow(binaryImage_median, cmap='gray')
    ax4.set_title('Median Image')

    # Plot the signal plots
    ax5 = plt.subplot(gs[1, :])
    ax5.plot(t_mean, sig_out_mean)
    ax5.set_title('Mean filter Plot')

    ax6 = plt.subplot(gs[2, :])
    ax6.plot(t_canny, sig_out_canny)
    ax6.set_title('Canny filter Plot')

    ax7 = plt.subplot(gs[3, :])
    ax7.plot(t_median, sig_out_median)
    ax7.set_title('Median filter Plot')

    plt.tight_layout()
    plt.show()



def plot_peaks_and_original(gray_im, sig_in, min_idx, max_idx, id):
    """
    Plot a cropped image together with its corresponding signal and detected peaks.

    This function displays:
        - The cropped grayscale image.
        - The original signal.
        - A duplicated “smoothed signal” line (as placeholder for actual smoothing).
        - Scatter markers for detected maxima (peaks) and minima (lows).

    Parameters
    ----------
    gray_im : ndarray
        The cropped grayscale image corresponding to the analyzed signal.
    sig_in : array-like
        The 1D signal extracted from the image or processing step.
    min_idx : array-like
        Indices of detected minima in the signal.
    max_idx : array-like
        Indices of detected maxima (peaks) in the signal.
    id : str or int
        Identifier for the image/sample, displayed in the plot title.

    Notes
    -----
    - Peaks are plotted in red; minima in blue.
    - The signal is plotted twice: once as the original and once as a dashed line
      labeled as “Smoothed Signal”. This placeholder can be replaced with a real
      smoothed signal if desired.
    - Uses a 2×1 grid layout.

    """

    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(2, 1)

    # Plot the image
    ax1 = plt.subplot(gs[0, :])
    ax1.imshow(gray_im, cmap='gray')
    ax1.set_title('Cropped Image: ' + str(id))

    # Plot the signal and peaks
    ax2 = plt.subplot(gs[1, :])
    ax2.plot(sig_in, label='Original Signal')
    ax2.plot(sig_in, label='Smoothed Signal', linestyle='--')
    ax2.scatter(max_idx, sig_in[max_idx], color='red', label='Peaks')
    ax2.scatter(min_idx, sig_in[min_idx], color='blue', label='Lows')
    ax2.legend()
    ax2.set_title('Filtered Signal and Peaks')

    plt.tight_layout()
    plt.show()



def fourier_interpolation(y, n_points):
    """
    Perform Fourier interpolation on a 1D signal.

    This method upsamples a signal by:
        1. Computing its Fourier transform.
        2. Zero-padding the Fourier coefficients (increasing frequency resolution).
        3. Applying an inverse FFT to obtain a smoothly interpolated signal.

    Fourier interpolation is especially useful for:
        - Smooth periodic signals
        - Avoiding interpolation artifacts introduced by spatial-domain methods

    Parameters
    ----------
    y : ndarray
        Original 1D input signal.
    n_points : int
        Desired number of samples in the interpolated signal.
        If `n_points` is smaller than the signal length, it is automatically
        increased to `2 * N` (where N = len(y)).

    Returns
    -------
    y_interp : ndarray
        The interpolated 1D signal (real-valued).

    Notes
    -----
    - Zero-padding in the frequency domain increases time-domain resolution.
    - Even- and odd-length signals are handled separately to preserve symmetry.
    - Output is normalized by `(n_points / N)` to preserve amplitude.
    """

    N = len(y)

    # Ensure minimum length
    if n_points < N:
        print("n_points is less than the length of the original signal. Setting n_points = 2*N.")
        n_points = 2 * N

    # Fourier transform
    Y = fft(y)

    # Zero-pad the Fourier coefficients
    if N % 2 == 0:  # even-length signal
        half = N // 2
        Y_interp = np.concatenate([
            Y[:half],
            np.zeros(n_points - N, dtype=complex),
            Y[half:]
        ])
    else:  # odd-length signal
        half = (N + 1) // 2
        Y_interp = np.concatenate([
            Y[:half],
            np.zeros(n_points - N, dtype=complex),
            Y[half:]
        ])

    # Inverse FFT and proper scaling
    y_interp = ifft(Y_interp) * (float(n_points) / N)

    return y_interp.real


def plot_y_interp_as_image(y_interp, width=500, height=300):
    """
    Converts the interpolated y-values into a binary image.

    Parameters:
        y_interp : NumPy array of interpolated y-values.
        width : Width of the output image.
        height : Height of the output image.

    Returns:
        img : Binary image with the interpolated waveform drawn.
    """
    # Create a blank black image
    img = np.zeros((height, width), dtype=np.uint8)

    # Normalize y-values to the image height (flip y-axis: higher y -> lower pixel row)
    y_interp_scaled = np.interp(y_interp, (y_interp.min(), y_interp.max()), (height - 1, 0)).astype(int)

    # Generate x-coordinates evenly spaced across the image width
    x_coords = np.linspace(0, width - 1, len(y_interp)).astype(int)

    # Draw the waveform as a continuous line
    for i in range(len(x_coords) - 1):
        cv2.line(img, (x_coords[i], y_interp_scaled[i]),
                 (x_coords[i + 1], y_interp_scaled[i + 1]), 255, 1)
    return img

def read_im_find_contour(queue, gray_im, zero_line, pxl_AMP):
    """
    Process a grayscale Doppler waveform image using three different filtering
    pipelines (mean thresholding, Canny edges, adaptive median), extract the
    upper contour in each method, smooth and normalize the resulting signals,
    and return them through a multiprocessing queue.

    This function performs:
        1. Adaptive mean thresholding → binary mask → upper contour extraction.
        2. Canny edge detection → contour extraction.
        3. Adaptive median thresholding → binary mask → contour extraction.

    For each method:
        - The top-most white pixel per column is identified.
        - Missing values are filled using fallback logic.
        - The contour is flipped vertically (waveform orientation).
        - Amplitude is normalized using pixel resolution.
        - Gaussian and Butterworth filters smooth the waveform.
        - A corresponding time axis is generated.

    Parameters
    ----------
    queue : multiprocessing.Queue
        Queue used to return results to the parent process.
    gray_im : ndarray
        Grayscale input image containing the Doppler waveform.
    zero_line : int
        Pixel row representing the Doppler baseline (zero velocity line).
    pxl_AMP : float
        Pixel-to-amplitude conversion factor (pixels per unit velocity).

    Returns
    -------
    Places into the queue a tuple in the following order:

        (
            sig_out_canny,      # smoothed Canny-based contour signal
            gray_im,            # original grayscale image
            binaryImage_mean,   # binary image from adaptive mean threshold
            Canny_img,          # Canny edge image
            binaryImage_median, # binary image from adaptive median threshold
            sig_out_mean,       # smoothed Mean-threshold contour signal
            t_mean,             # time axis for mean filter signal
            sig_out_canny,      # smoothed Canny contour signal (duplicate)
            t_canny,            # time axis for Canny signal
            sig_out_median,     # smoothed median-threshold contour signal
            t_median            # time axis for median filtering signal
        )

    Notes
    -----
    - Contour extraction is performed column-by-column by locating the first
      white pixel from the top. If none is found, the function falls back to
      using the last white pixel anywhere in the image.
    - Butterworth filtering parameters:
          order = 2, cutoff = 0.2 (normalized)
    - Gaussian filter smoothing is applied before Butterworth filtering.
    - The function assumes the waveform appears as a bright structure on
      a dark background.
    - The result contains THREE versions of the extracted waveform:
          - Mean filter contour
          - Canny contour
          - Adaptive median contour

    """
    ###################################
    # Mean filter and binary conversion
    binaryImage_mean = cv2.adaptiveThreshold(
        gray_im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0
    )

    binaryImage_mean_gray = (binaryImage_mean > 25).astype(np.uint8) * 255
    full_frame = binaryImage_mean_gray.astype(float)

    size_full_frame = full_frame.shape
    lfull_frame = size_full_frame[1]
    wfull_frame = size_full_frame[0]

    row_sums = np.sum(full_frame, axis=1)
    max_AMP_row = np.argmax(row_sums > 2)
    max_AMP = abs(max_AMP_row - zero_line) / pxl_AMP

    # Extract contour from mean-thresholded image
    v = np.zeros(lfull_frame)
    for j in range(lfull_frame):
        vv = full_frame[:, j]
        idx = np.any(vv == 255)
        if idx:
            v[j] = np.where(vv == 255)[0][0]
        else:
            idx2 = np.any(full_frame == 255, axis=1)
            if np.any(idx2):
                v[j] = np.where(idx2)[0][-1]

    # Smooth the contour
    vnew = wfull_frame - v
    vnew = (max_AMP / np.max(vnew)) * vnew
    fvnew = gaussian_filter(vnew, sigma=2)
    b, a = butter(2, 0.2, 'low')
    sig_out_mean = filtfilt(b, a, fvnew)
    t_mean = np.arange(len(vnew))

    ###################################
    # Canny filter
    Canny_img = (canny(gray_im, sigma=0.8,
                       low_threshold=72, high_threshold=128) * 255).astype(np.uint8)

    full_frame_canny = Canny_img.astype(float)

    size_full_frame = full_frame_canny.shape
    lfull_frame = size_full_frame[1]
    wfull_frame = size_full_frame[0]

    # Extract contour from Canny edges
    f = np.zeros(lfull_frame)
    for j in range(lfull_frame):
        v_canny = full_frame_canny[:, j]
        idx_canny = np.any(v_canny == 255)
        if idx_canny:
            f[j] = np.where(v_canny == 255)[0][0]
        else:
            idx2_canny = np.any(full_frame_canny == 255, axis=1)
            f[j] = np.where(idx2_canny)[0][-1]

    vnew2 = wfull_frame - f
    vnew2 = (max_AMP / np.max(vnew2)) * vnew2
    fvnew2 = gaussian_filter(vnew2, sigma=4)
    sig_out_canny = filtfilt(b, a, fvnew2)
    t_canny = np.arange(len(fvnew2))

    ###################################
    # Adaptive median detection
    binaryImage_median = 255 * (
        gray_im > filters.threshold_local(gray_im, 15, method='median', mode='reflect')
    )

    binaryImage_median_gray = (binaryImage_median > 25).astype(np.uint8)
    full_frame_median = binaryImage_median_gray.astype(float)

    size_full_frame_median = full_frame_median.shape
    lfull_frame = size_full_frame_median[1]
    wfull_frame = size_full_frame_median[0]

    row_sums = np.sum(full_frame_median, axis=1)
    max_AMP_row = np.argmax(row_sums > 2)
    max_AMP = abs(max_AMP_row - zero_line) / pxl_AMP

    v_median = np.zeros(lfull_frame)
    for j in range(lfull_frame):
        vv_median = full_frame_median[:, j]
        idx_median = np.any(vv_median == 1)
        if idx_median:
            v_median[j] = np.where(vv_median == 1)[0][0]
        else:
            idx2_median = np.any(full_frame_median == 1, axis=1)
            if np.any(idx2_median):
                v_median[j] = np.where(idx2_median)[0][-1]

    vnew_median = wfull_frame - v_median
    vnew_median2 = (max_AMP / np.max(vnew_median)) * vnew_median
    fvnew_median = gaussian_filter(vnew_median2, sigma=4)
    sig_out_median = filtfilt(b, a, fvnew_median)
    t_median = np.arange(len(fvnew_median))

    ###################################

    res1 = (
        sig_out_canny, gray_im, binaryImage_mean, Canny_img,
        binaryImage_median, sig_out_mean, t_mean,
        sig_out_canny, t_canny, sig_out_median, t_median
    )

    queue.put(res1)


def plot_freq_explorer(f, P1):
    """Plot a single-sided amplitude spectrum."""
    plt.figure(figsize=(8, 5))
    plt.stem(f, P1, use_line_collection=True)
    plt.title('Single-Sided Amplitude Spectrum of X(t)')
    plt.xlabel('f (Hz)')
    plt.ylabel('|P1(f)|')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



def find_top_three_indices(arr):
    """
    Return the top 3 values and their indices, sorted by index (ascending).
    If arr has fewer than 3 elements, returns all available.
    """
    arr = np.asarray(arr)

    # Get valid indices (ignore NaNs)
    valid = ~np.isnan(arr)
    arr_valid = arr[valid]
    idx_valid = np.where(valid)[0]

    if arr_valid.size == 0:
        return np.array([]), np.array([])

    # Number of peaks to return
    k = min(3, arr_valid.size)

    # Top k values
    sorted_indices = np.argsort(arr_valid)
    top_k_local = sorted_indices[-k:]

    # Convert local indices back to original array indices
    top_k_global = idx_valid[top_k_local]

    # Sort by *global index*
    order = np.argsort(top_k_global)

    return arr[top_k_global][order], top_k_global[order]



def freq_explorer(queue, sig, fs=100, frame_flag=False, pre_emphasis_alpha=0):
    """
    Analyze the frequency content of a 1D signal and extract key spectral features.

    This function:
        1. Removes the DC component from the input signal.
        2. Optionally applies a pre-emphasis filter in the frequency domain.
        3. Computes the FFT and single-sided amplitude spectrum.
        4. Determines the dominant frequency (Heart Rate in bpm).
        5. Removes the dominant frequency and its harmonics from the spectrum.
        6. Identifies the next top three spectral peaks.
        7. Returns all results via a multiprocessing queue.

    Parameters
    ----------
    queue : multiprocessing.Queue
        Queue used to return the results to the parent process.
    sig : array-like
        Input 1D signal for frequency analysis.
    fs : float, optional
        Sampling frequency in Hz. Default is 100 Hz.
    frame_flag : bool, optional
        Currently unused flag; reserved for future processing options. Default is False.
    pre_emphasis_alpha : float, optional
        Pre-emphasis filter coefficient. If 0, no pre-emphasis is applied. Default is 0.

    Returns
    -------
    None
        Results are placed into the `queue` as a tuple:
        (
            sig_fft,        # FFT of the signal after DC removal and optional pre-emphasis
            dc_comp,        # DC component of the original signal
            HR,             # Dominant frequency converted to Heart Rate (bpm)
            top3_f_values,  # Amplitudes of the next top 3 peaks (after removing HR and harmonics)
            top3_f_index,   # Frequencies corresponding to top3_f_values
            f,              # Frequency vector corresponding to single-sided spectrum
            P1              # Single-sided amplitude spectrum of the signal
        )

    Notes
    -----
    - The dominant frequency is considered the Heart Rate.
    - Harmonics of the dominant frequency are zeroed before extracting the top 3 secondary peaks.
    - Pre-emphasis is applied via convolution in the frequency domain if alpha > 0.
    - The function does not return values directly; results are only sent through the queue.
    """
    # Sampling frequency
    T = 1 / fs  # Sampling period
    sig = sig[~np.isnan(sig)]
    L = len(sig)  # Length of signal
    # t = np.arange(0, L) * T  # Time vector

    dc_comp = np.mean(sig)  # DC calculation
    sig_fft = np.fft.fft(sig - dc_comp)  # Removal of DC component

    # Use pre-emphasis filter only if alpha is specified
    if pre_emphasis_alpha != 0:
        # Pre-emphasis filter parameter
        h = [1, -pre_emphasis_alpha]  # Frequency response of the filter
        pre_emphasized = np.convolve(sig_fft, h, mode='same')
        sig_fft = pre_emphasized

    P2 = np.abs(sig_fft / L)
    P1 = P2[:(L // 2) + 1]
    P1[1:-2] = 2 * P1[1:-2]

    f = fs * np.arange(0, (L // 2) + 1) / L

    # The main frequency is the Heart Rate
    # Find the index of the maximum value
    max_index = np.argmax(P1)
    HR = 60 * f[max_index]  # Heart rate in bpm (beats per minute)

    subvector_indices = np.arange(0, len(P1), max_index)
    subvector_indices = subvector_indices[subvector_indices <= max_index * 10]

    P3 = P1.copy()  # the fft without the HR and its echos
    P3[subvector_indices] = 0

    top3_f_values, top3_f_inx = find_top_three_indices(P3)
    top3_f_index = [f[top3_f_inx[0]], f[top3_f_inx[1]], f[top3_f_inx[2]]]

    res = (sig_fft, dc_comp, HR, top3_f_values, top3_f_index, f, P1)

    queue.put(res)



def plot_peaks(sig_in, max_idx, min_idx):
    """
    Plot a signal with its peaks and lows.

    Parameters
    ----------
    sig_in : array-like
        Input signal.
    max_idx : array-like
        Indices of peaks.
    min_idx : array-like
        Indices of lows.
    """
    plt.figure()
    plt.plot(sig_in, label='Original Signal')
    plt.scatter(max_idx, sig_in[max_idx], color='red', label='Peaks')
    plt.scatter(min_idx, sig_in[min_idx], color='blue', label='Lows')
    plt.legend()
    plt.title('Smoothed Signal with Peaks and Lows')
    plt.show()



def smooth_between_extremes(signal, peak_indices, low_indices):
    """
    Smooth a signal between peaks and lows using cubic spline interpolation.

    Parameters
    ----------
    signal : array-like
        Input signal to smooth.
    peak_indices : array-like
        Indices of peaks (maxima).
    low_indices : array-like
        Indices of lows (minima).

    Returns
    -------
    smoothed_signal : array
        Signal after smoothing between extremes.
    """
    all_extremes = np.sort(np.concatenate((peak_indices, low_indices)))
    smoothed_signal = np.copy(signal)

    for i in range(len(all_extremes) - 1):
        start_index = all_extremes[i]
        end_index = all_extremes[i + 1]
        # Create a cubic spline interpolation between extremes
        x = np.arange(start_index, end_index + 1)
        y = signal[start_index:end_index + 1]
        cs = CubicSpline(x, y)
        # Replace original signal values between extremes with smoothed values
        smoothed_signal[start_index:end_index + 1] = cs(x)

    return smoothed_signal


def filter_peaks_and_lows_by_distance(peak_indices, peak_values, low_indices, low_values, min_distance):
    """
    Filter peaks and lows to ensure a minimum distance between them.

    Parameters
    ----------
    peak_indices : array-like
        Indices of detected peaks.
    peak_values : array-like
        Values of detected peaks.
    low_indices : array-like
        Indices of detected lows.
    low_values : array-like
        Values of detected lows.
    min_distance : int
        Minimum allowed distance between consecutive peaks or lows.

    Returns
    -------
    filtered_peak_indices : array
        Indices of peaks after filtering.
    filtered_peak_values : array
        Values of peaks after filtering.
    filtered_low_indices : array
        Indices of lows after filtering.
    filtered_low_values : array
        Values of lows after filtering.
    """
    # Filter peaks
    filtered_peak_indices = [peak_indices[0]]
    filtered_peak_values = [peak_values[0]]
    for i in range(1, len(peak_indices)):
        if peak_indices[i] - peak_indices[i-1] >= min_distance:
            filtered_peak_indices.append(peak_indices[i])
            filtered_peak_values.append(peak_values[i])
        else:
            # If peaks are closer than min_distance, keep the higher one
            if peak_values[i] > peak_values[i-1]:
                filtered_peak_values[-1] = peak_values[i]
                filtered_peak_indices[-1] = peak_indices[i]

    # Filter lows
    filtered_low_indices = [low_indices[0]]
    filtered_low_values = [low_values[0]]
    for i in range(1, len(low_indices)):
        if low_indices[i] - low_indices[i-1] >= min_distance:
            filtered_low_indices.append(low_indices[i])
            filtered_low_values.append(low_values[i])
        else:
            # If lows are closer than min_distance, keep the lower one
            if low_values[i] < low_values[i-1]:
                filtered_low_values[-1] = low_values[i]
                filtered_low_indices[-1] = low_indices[i]

    return np.array(filtered_peak_indices), np.array(filtered_peak_values), np.array(filtered_low_indices), np.array(filtered_low_values)




def segmentor(sig, min_idx):
    """
    Segment a signal between consecutive minima.

    Parameters
    ----------
    sig : array-like
        Input signal to segment.
    min_idx : array-like
        Indices of minima in the signal.

    Returns
    -------
    segments : list of dicts
        Each dict contains:
            'segment' : signal values between two minima
            'segment_idx' : corresponding indices in the original signal
    """
    n = len(min_idx) - 1  # number of anticipated segments
    segments = []

    for i in range(n):
        segment = sig[min_idx[i]:min_idx[i+1]+1]
        segment_idx = list(range(min_idx[i], min_idx[i+1]+1))

        segments.append({'segment': segment, 'segment_idx': segment_idx})

    return segments



def legacy_features_calculator(max_val, min_val, dc_comp):
    """
    This function calculates the Pulsatility Index (PI), Systole/Diastole ratio (SD),
    and the Resistance Index (RI).

    Parameters:

        max_val: Systole peaks
        min_val: Diastole low-peaks
        dc_comp: Mean value of the signal, also the DC component of the signal

    Returns:
        PI: Pulsatility Index
        SD: Systole/Diastole ratio
        RI: Resistance Index
    """
    PI = (max_val - min_val[1:]) / dc_comp
    SD = max_val / min_val[1:]
    RI = (max_val - min_val[1:]) / max_val

    return PI, SD, RI


def dvd_s_calculator(fs, segments, a_100):
    """
    This function calculates the DVD_S (Doppler Velocity Discontinuity - Spectral).

    Parameters:
        fs: Sampling frequency
        segments: List of segments
        a_100: A at 100% of the cycle,i.e, next MINVAL

    Returns:
        DVD_S: Doppler Velocity Discontinuity - Spectral
    """
    n = len(segments)
    m = [len(seg['segment']) for seg in segments]
    average_len = sum(m) / n

    # Identifying the last 10% of the frame
    a_90 = [seg['segment'][-round(average_len * 0.1)] for seg in segments]

    delta_t = (average_len - 0.1 * average_len) / fs  # t_100% - t_90% of the duty cycle

    DVD_S = (a_100[1:] - a_90) / delta_t

    return DVD_S


def decay_index(segment, fs):
    """
    This function fits an exponential curve to the provided signal segment and estimates the Decay Index (DI).

    Parameters:
        segment: Signal segment
        fs: Sampling frequency

    Returns:
        DI_fitted: Fitted Decay Index
    """
    # Extract A, V, and t from the signal and provided time vector
    A = np.max(segment)
    max_index = np.argmax(segment)
    V = segment[max_index:]
    t = np.arange(len(V)) / fs

    # Define the model function (exponential curve)
    model = lambda t, DI: A * np.exp(-DI * t)

    # Initial guess for DI
    initial_guess_DI = 5.0

    # Fit the model to the data, estimating DI
    popt, _ = curve_fit(model, t, V, p0=initial_guess_DI)

    DI_fitted = popt[0]

    return DI_fitted


def calculate_duty_cycle(min_idx, max_idx):
    """
    Calculate the average duty cycle of systole in a cardiac cycle.

    Parameters
    ----------
    min_idx : array-like
        Indices of minima (start of cardiac cycles).
    max_idx : array-like
        Indices of maxima (systolic peaks).

    Returns
    -------
    float
        Average duty cycle of systole (ratio of systole duration to total cycle duration).
    """
    # Total duration of the cardiac cycle in samples (not time)
    T_systole = max_idx - min_idx[:-1]
    T_diastole = min_idx[1:] - max_idx
    T_total = T_systole + T_diastole

    # Duty cycle for systole
    duty_cycle_systole = T_systole / T_total #(ratio)

    # Duty cycle for diastole
    #duty_cycle_diastole = T_diastole / T_total

    return np.mean(duty_cycle_systole)


def Framer(signal_contour, MinIdx, n, m):
    """
    Divide a signal into overlapping frames based on minima indices.

    Parameters
    ----------
    signal_contour : array-like
        Input signal to frame.
    MinIdx : array-like
        Indices of minima used to define frame boundaries.
    n : int
        Number of minima per frame.
    m : int
        Step size in minima between consecutive frames.

    Returns
    -------
    frames : list of arrays
        List of signal segments (frames) extracted from the input signal.
    """
    frames = []
    i = 0
    while i * m + n < len(MinIdx):
        start = MinIdx[i * m]
        end = MinIdx[i * m + n]
        frames.append(signal_contour[start:end])
        i += 1
    return frames


def calculate_moving_averages_for_multiple_lists(features_lists, additional_arrays_list, frames_num, n, m):
    """
    Compute moving averages for multiple lists of features and additional arrays.

    Parameters
    ----------
    features_lists : list of lists of arrays
        Legacy features for which averages are calculated per array.
    additional_arrays_list : list of arrays
        Additional features to compute moving averages over.
    frames_num : int
        Number of frames to compute averages for.
    n : int
        Window size (number of elements) for moving average.
    m : int
        Step size for moving the window between frames.

    Returns
    -------
    all_moving_averages : ndarray
        Combined array of legacy averages and moving averages for additional arrays.
    """
    # Calculate the average for each ndarray within each inner list
    legacy_averages = [[np.mean(arr) for arr in inner_list] for inner_list in features_lists]
    legacy_averages = np.array(legacy_averages) # convert to ndarray

    moving_averages = np.zeros((frames_num, len(additional_arrays_list)))

    # Append moving averages for additional arrays
    j = 0
    for additional_feature in additional_arrays_list:
        cumsum = np.cumsum(additional_feature)
        k = 0
        min_ftr_idx = 0
        for i in range(0, frames_num):  # each iteration is 1 frame
            if min_ftr_idx + n - 1 < len(cumsum):  # Ensure index is within bounds
                window_sum = (cumsum[min_ftr_idx + n - 1] - cumsum[min_ftr_idx - 1]
                              if min_ftr_idx > 0 else cumsum[min_ftr_idx + n - 1])
                moving_avg = window_sum / n
                moving_averages[k][j] = moving_avg
            else:
                # Handle the case where the index goes out of bounds, maybe set to NaN or 0
                moving_averages[k][j] = np.nan  # Or another appropriate value
            min_ftr_idx += m
            k += 1
        j += 1

    all_moving_averages = np.concatenate((legacy_averages, moving_averages), axis=1)

    return all_moving_averages


def Integrator(id, gw, gw_in_days, legacy_features_frames, DVD_S, DI, duty_cycle_frames, frames, dc_comp, HR, fs,
               gw_delivery, gw_delivery_in_days,
               top3_f_values, top3_f_indices, n, m, file_path=fr'C:\Python_Projects\pythonProject\features.csv'):
    """
    Combine multiple signal features into a DataFrame and save to CSV.

    Parameters
    ----------
    id : str
        Patient or sample identifier.
    gw : int
        Gestational week.
    gw_in_days : int
        Gestational age in days.
    legacy_features_frames : list of lists
        Legacy features for each frame.
    DVD_S, DI : array-like
        Additional features to include.
    duty_cycle_frames : array-like
        Duty cycle values for each frame.
    frames : list
        Frame indices or segments.
    dc_comp : float
        Mean amplitude of the signal.
    HR : float
        Heart rate in bpm.
    fs : int
        Sampling frequency.
    gw_delivery : int
        Gestational week at delivery.
    gw_delivery_in_days : int
        Gestational age at delivery in days.
    top3_f_values : array-like
        Amplitudes of top three frequencies.
    top3_f_indices : array-like
        Frequencies corresponding to top three amplitudes.
    n, m : int
        Window size and step for moving averages.
    file_path : str, optional
        Path to save the CSV file (default: 'C:\\Python_Projects\\pythonProject\\features.csv').

    Returns
    -------
    None
        Saves the compiled features to CSV.
    """

    # Combine features into a matrix
    additional_features = np.vstack([DVD_S, DI])

    if gw <= 12:
        trimester = 1
    elif gw <= 27:
        trimester = 2
    else:  # gw >= 28
        trimester = 3

    # Calculate moving means for each feature
    features_moving_averages = calculate_moving_averages_for_multiple_lists(
        legacy_features_frames, additional_features, len(frames), n, m
    )

    df = pd.DataFrame()

    df['frame_id'] = [i for i in range(1, len(features_moving_averages) + 1)]
    df['id_int'] = int(''.join(filter(str.isdigit, id)))
    df['id'] = id
    df['id_frame_id'] = df['id'].astype(str) + '_' + df['frame_id'].astype(str)
    df = df.drop(['frame_id'], axis=1)

    df['PI'] = [sublist[0] for sublist in features_moving_averages]
    df['SD'] = [sublist[1] for sublist in features_moving_averages]
    df['RI'] = [sublist[2] for sublist in features_moving_averages]
    df['DVD_S'] = [sublist[3] for sublist in features_moving_averages]
    df['DI'] = [sublist[4] for sublist in features_moving_averages]
    df['duty_cycle'] = duty_cycle_frames
    df['trimester'] = trimester
    df['gw'] = gw
    df['gw_in_days'] = gw_in_days
    df['mean_amp'] = dc_comp
    df['HR'] = HR
    df['top3_f_amp'] = top3_f_values
    df['top3_f_freq'] = top3_f_indices
    df['frames'] = frames
    df['gw_delivery'] = gw_delivery
    df['gw_delivery_in_days'] = gw_delivery_in_days
    df['fs'] = fs

    verify_column_order = [
        'id', 'id_frame_id', 'id_int', 'trimester', 'gw', 'gw_in_days',
        'gw_delivery', 'gw_delivery_in_days', 'fs', 'PI', 'SD', 'RI',
        'DVD_S', 'DI', 'mean_amp', 'duty_cycle', 'HR',
        'top3_f_freq', 'top3_f_amp', 'frames'
    ]

    df = df[verify_column_order]

    if os.path.exists(file_path):
        # If the file exists, append to it
        df.to_csv(file_path, mode='a', index=False, header=False)
    else:
        # If the file does not exist, create it and write the DataFrame to it
        df.to_csv(file_path, index=False)


def Integrator_aug(row, legacy_features_frames, DVD_S, DI, duty_cycle_fr, dc_comp, HR,
                   top3_f_values, top3_f_indices, n, m, file_path=fr'C:\Python_Projects\pythonProject\features.csv'):
    """
    Update a single DataFrame row with computed features and save to CSV.

    Parameters
    ----------
    row : pandas.DataFrame
        Single-row DataFrame to be updated with features.
    legacy_features_frames : list of lists
        Legacy features for the frame.
    DVD_S, DI : array-like
        Additional features to include.
    duty_cycle_fr : float or array-like
        Duty cycle value(s) for the frame.
    dc_comp : float
        Mean amplitude of the signal.
    HR : float
        Heart rate in bpm.
    top3_f_values : array-like
        Amplitudes of top three frequencies.
    top3_f_indices : array-like
        Frequencies corresponding to top three amplitudes.
    n, m : int
        Window size and step for moving averages.
    file_path : str, optional
        Path to save the CSV file (default: 'C:\\Python_Projects\\pythonProject\\features.csv').

    Returns
    -------
    None
        Updates the DataFrame row and saves it to CSV.
    """
    # Combine features into a matrix
    additional_features = np.vstack([DVD_S, DI])

    ###########
    # Calculate moving means for each feature
    features_moving_averages = calculate_moving_averages_for_multiple_lists(legacy_features_frames,
                                                                            additional_features, 1, n, m)

    row['PI'] = [sublist[0] for sublist in features_moving_averages]
    row['SD'] = [sublist[1] for sublist in features_moving_averages]
    row['RI'] = [sublist[2] for sublist in features_moving_averages]
    row['DVD_S'] = [sublist[3] for sublist in features_moving_averages]
    row['DI'] = [sublist[4] for sublist in features_moving_averages]
    row['duty_cycle'] = duty_cycle_fr
    row['mean_amp'] = dc_comp
    row['HR'] = HR
    row['top3_f_amp'] = [top3_f_values]
    row['top3_f_freq'] = [top3_f_indices]
    verify_column_order = ['id', 'id_frame_id', 'id_int', 'trimester', 'gw',
                           'gw_in_days', 'gw_delivery', 'gw_delivery_in_days', 'fs', 'PI', 'SD', 'RI', 'DVD_S', 'DI',
                           'mean_amp', 'duty_cycle', 'HR', 'top3_f_freq', 'top3_f_amp', 'frames']

    row = row[verify_column_order]

    if os.path.exists(file_path):
        # If the file exists, append to it
        row.to_csv(file_path, mode='a', index=False, header=False)
    else:
        # If the file does not exist, create it and write the DataFrame to it
        row.to_csv(file_path, index=False)


def sniplet(id, rectcrop):
    """
    Crop a region from an image and save it as a new file.

    Parameters
    ----------
    id : str
        Identifier of the image file (without extension).
    rectcrop : tuple/list of 4 ints
        Cropping coordinates (start_row, end_row, start_col, end_col).

    Returns
    -------
    None
        Saves the cropped image to disk.
    """
    path = fr'C:\Python_Projects\DB\{id}.jpg'  # Use raw string for file path
    full_img = cv2.imread(path)
    snipped_imp = full_img[rectcrop[0]: rectcrop[1], rectcrop[2]: rectcrop[3]]
    output_path = fr'C:\Python_Projects\snipped_db\{id}_sn.jpg'
    cv2.imwrite(output_path, snipped_imp)

    # Optionally, you can print a confirmation message
    print(f"{id} Snipplet saved to {output_path}")



# Define the augmentation functions
def add_noise(series, noise_level=0.005):
    """
    Add Gaussian noise to a time series.

    Parameters
    ----------
    series : array-like
        Input time series to be augmented.
    noise_level : float, optional
        Base standard deviation for Gaussian noise (default 0.005).

    Returns
    -------
    array-like
        Time series with added Gaussian noise.
    """
    noise_level = np.random.uniform(0.002, 0.005)  # Random noise factor between 0.002 and 0.005
    noise = np.random.normal(loc=0.0, scale=noise_level * np.std(series), size=series.shape)
    return series + noise


def scaling(series, scale_factor=0.05):
    """
    Randomly scale a time series around its original amplitude.

    Parameters
    ----------
    series : array-like
        Input time series to be scaled.
    scale_factor : float, optional
        Base scaling factor range (default 0.05).

    Returns
    -------
    array-like
        Scaled time series.
    """
    scale_factor = np.random.uniform(0.95, 1.05)  # Random scaling factor between 0.95 and 1.05
    scale = np.random.uniform(1 - scale_factor, 1 + scale_factor)
    return series * scale


def time_warp(series, warp_factor=0.05):
    """
    Slightly warp the time steps of a series.

    Parameters
    ----------
    series : array-like
        Input time series to be warped.
    warp_factor : float, optional
        Degree of temporal distortion (default 0.05).

    Returns
    -------
    array-like
        Time-warped time series.
    """
    time_steps = np.linspace(0, 1, num=len(series))
    max_warp = warp_factor * np.std(time_steps)
    warped_time_steps = np.interp(time_steps + np.random.normal(-max_warp, max_warp, len(time_steps)), time_steps, series)
    return np.interp(time_steps, np.sort(warped_time_steps), series)


def apply_random_augmentation(series):
    """
    Apply a randomly chosen augmentation (noise or scaling) to a time series.

    Parameters
    ----------
    series : array-like
        Input time series to augment.

    Returns
    -------
    array-like
        Augmented time series.
    """
    augmentation_methods = {
        'add_noise': add_noise,
        'scaling': scaling,
    }
    method = random.choice(list(augmentation_methods.keys()))
    series_out = augmentation_methods[method](series)
    return series_out


def resample_signal(signal, original_fs, target_fs):
    """
    Resample a 1D signal to a new sampling frequency.

    Parameters
    ----------
    signal : array-like, list, or pandas.Series
        Input signal to resample.
    original_fs : float
        Original sampling frequency of the signal.
    target_fs : float
        Desired sampling frequency.

    Returns
    -------
    numpy.ndarray
        Resampled 1D signal.
    """
    # Convert signal to a NumPy array if it's a pandas Series
    if isinstance(signal, pd.Series):
        signal = signal.values
    elif isinstance(signal, list):
        signal = np.array(signal)
    elif not isinstance(signal, np.ndarray):
        print(f"Unexpected signal type: {type(signal)}")
        raise ValueError("Signal must be a pandas Series, list, or numpy array")

    # Ensure the signal is a 1D numpy array
    signal = np.asarray(signal)

    # Calculate the number of samples in the resampled signal
    num_samples = int(len(signal) * target_fs / original_fs)

    # Resample the signal
    resampled_signal = resample(signal, num_samples)

    return resampled_signal