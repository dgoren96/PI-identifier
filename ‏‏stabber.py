"""
This script loads Doppler ultrasound images, allows the user to manually crop
regions of interest (ROI), extracts the zero-line, amplitude scaling, and
time scaling parameters, and stores the processed information into stabs.csv.

Main steps:
-----------
1. Load gestational week (gw) table and match IDs.
2. Load each image, crop dynamically based on user-selected ROI.
3. Detect zero-line using statistical scoring inside the cropped ROI.
4. Ask user to click reference amplitude line (45 / 60 / manual).
5. Ask user to click time boundaries for fs calculation.
6. Store outputs per ID:
      - fs (time spacing)
      - pxl_AMP (pixel-to-AMP scaling)
      - initial_zero_line (detected zero-line)
      - rectcrop (cropped rectangle coordinates)
7. Append results to stabs.csv

User Interaction:
-----------------
The user selects ROI and clicks specific reference locations:
- ROI selection (zero-line search area)
- Y-axis reference line (45/60/manual)
- Two boundary clicks for fs calculation

Dependencies:
-------------
cv2, numpy, pandas, matplotlib, scipy, re, os
Custom functions from assisting_functions.py:
- get_original_coordinates
- display_image_with_cross_cursor
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import assisting_functions as af
import multiprocessing
import pandas as pd
from scipy.signal import find_peaks
import os
import re

make_plots_flag = False  # Global flag for enabling/disabling plots

# -------------------------------------------------------------------------
# ID selection (commented presets kept for manual switching)
# -------------------------------------------------------------------------

#ID = [...]
ID = ['132a']  # <-- ACTIVE ID LIST

# -------------------------------------------------------------------------
# Example graph (not necessary for pipeline; left for reference)
# -------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.set_xlim(0, 5500)
ax.set_ylim(0, 1.4)
ax.set_xticks(np.arange(0, 5501, 500))
ax.set_xticks(np.arange(0, 5501, 100), minor=True)
ax.set_yticks(np.arange(0, 1.41, 0.2))
ax.set_yticks(np.arange(0, 1.41, 0.05), minor=True)
ax.grid(True, which="major", linewidth=0.8)
ax.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.6)
ax.set_xlabel(r"$r$ [m]")
ax.set_ylabel(r"$I$ [W]")
ax.set_title("Graph with Grid")
plt.show()

# -------------------------------------------------------------------------
# Load gestational age table and filter for selected IDs
# -------------------------------------------------------------------------

GW = pd.read_csv(r'C:\Python_Projects\pythonProject\gw.csv')

# Extract numeric part of IDs (e.g., "132a" -> 132)
id_int = [int(re.match(r'\d+', id).group()) for id in ID]

gw = GW[GW['id_int'].isin(id_int)]

fs = []
rectcrop = []
pxl_AMP = []
zero_line = []

df = pd.DataFrame(columns=gw.columns)
j = 0

# -------------------------------------------------------------------------
# Main Loop Over All IDs
# -------------------------------------------------------------------------
for id in ID:

    # ---------------------------------------
    # Retrieve gestational week from table
    # ---------------------------------------
    gw_row = gw[gw['id_int'] == id_int[j]].reset_index(drop=True)
    if not gw_row.empty:
        df.loc[j] = gw_row['gw'].iloc[0]
    else:
        print(f"No match found for id_int {id_int[j]}")

    df.loc[j, 'id'] = id

    # ---------------------------------------
    # Load image and apply Y_SKIP cropping
    # ---------------------------------------
    img_path = fr'C:\Python_Projects\DB\{id}.jpg'
    Y_SKIP = 400

    A_full = cv2.imread(img_path)
    A = A_full[Y_SKIP:, :]  # Remove top region

    height, width, _ = A.shape
    print("\n*************\nID:", id)

    # ---------------------------------------
    # ROI Selection - resized for better visibility
    # ---------------------------------------
    screen_res = (1320, 685)  # Adjust to match display resolution

    orig_height, orig_width = A.shape[:2]
    resized_image = cv2.resize(A, screen_res)

    width_ratio = screen_res[0] / orig_width
    height_ratio = screen_res[1] / orig_height

    cv2.namedWindow(fr'Select ROI including zero-line, id={id}', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(fr'Select ROI including zero-line, id={id}', *screen_res)
    cv2.putText(resized_image, str(id), (100, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow(fr'Select ROI including zero-line, id={id}', resized_image)

    resized_pixel_x1, resized_pixel_y1, resized_w, resized_h = cv2.selectROI(
        fr'Select ROI including zero-line, id={id}',
        resized_image, False
    )
    cv2.destroyAllWindows()

    # ---------------------------------------
    # Extract ROI for zero-line scoring
    # ---------------------------------------
    img3 = resized_image[
        resized_pixel_y1:resized_pixel_y1 + resized_h,
        resized_pixel_x1:resized_pixel_x1 + resized_w
    ]
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    median_values = np.median(img3, axis=1) / 255
    var_values = np.var(img3.astype(float), axis=1)
    diff_values = np.diff(img3.astype(float), axis=1)

    sc_var = 0.999 / (1 + var_values)
    sc_median = 0.001 * median_values
    score = sc_var + sc_median

    # Reject too-dark rows
    for i in range(len(score)):
        if median_values[i] < (140 / 255):
            score[i] = 0

    zero_line_id_new = np.argmax(score)

    # Map ROI back to original full-resolution coordinates
    original_x, original_y = af.get_original_coordinates(
        resized_pixel_x1, resized_pixel_y1,
        width_ratio, height_ratio
    )
    original_w = int(resized_w / width_ratio)
    original_h = int(resized_h / height_ratio)
    original_zero_line_id_new = int(zero_line_id_new / height_ratio)

    # rectcrop format: [y_start, y_end, x_start, x_end]
    rectout = [
        Y_SKIP + original_y,
        Y_SKIP + original_y + original_zero_line_id_new,
        original_x,
        original_x + original_w
    ]

    # ---------------------------------------
    # Verify cropped region visually
    # ---------------------------------------
    I2 = A_full[rectout[0]:rectout[1], rectout[2]:rectout[3]]
    af.display_image_with_cross_cursor(I2, 'Verify rectcrop_im - click to continue')

    # ---------------------------------------
    # Y-axis amplitude reference (pixel per AMP)
    # ---------------------------------------
    print('LEFT click = 45 | RIGHT = 60 | MIDDLE = manual')
    x, ref_line, event_type = af.display_image_with_cross_cursor(A, 'Please LEFT click the Y value on the "45", RIGHT click on the "60" line, or MIDDLE click for else : ')

    if event_type == 'left_click':
        reference_AMP = 45
    elif event_type == 'right_click':
        reference_AMP = 60
    else:
        reference_AMP = int(input("Enter the reference line value: "))

    pxl_AMP_id = abs(ref_line - zero_line_id_new) / reference_AMP

    # ---------------------------------------
    # Time spacing fs (pixel distance between boundaries)
    # ---------------------------------------
    plt.close('all')
    boundary1, y, _ = af.display_image_with_cross_cursor(A, "Click FIRST time boundary")
    plt.close('all')
    boundary2, y, _ = af.display_image_with_cross_cursor(A, "Click SECOND time boundary")

    fs_id = round(abs(boundary1 - boundary2), -1)  # Round to nearest 10

    # ---------------------------------------
    # Accumulate results
    # ---------------------------------------
    if id == ID[0]:
        rectcrop = [list(rectout)]
        zero_line = zero_line_id_new
        pxl_AMP = pxl_AMP_id
        fs = fs_id
    else:
        rectcrop.append(list(rectout))
        zero_line = np.vstack([zero_line, zero_line_id_new])
        pxl_AMP = np.vstack([pxl_AMP, pxl_AMP_id])
        fs = np.vstack([fs, fs_id])

    j += 1

# -------------------------------------------------------------------------
# Add extracted parameters to DataFrame
# -------------------------------------------------------------------------
df['fs'] = fs
df['pxl_AMP'] = pxl_AMP
df['initial_zero_line'] = zero_line
df['rectcrop'] = rectcrop

# Ensure correct column order
verify_column_order = ['id', 'gw', 'fs', 'pxl_AMP', 'initial_zero_line', 'rectcrop']
df = df[verify_column_order]

# -------------------------------------------------------------------------
# Save/append results to stabs.csv
# -------------------------------------------------------------------------
file_path = r'C:\Python_Projects\pythonProject\stabs.csv'

try:
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', index=False, header=False)
    else:
        df.to_csv(file_path, index=False, header=True)

except PermissionError:
    print("Permission denied. Close the file and press ENTER.")
    input()
    df.to_csv(file_path, mode='a', index=False, header=False)

print("Finished")
input("Press Enter to exit...")
