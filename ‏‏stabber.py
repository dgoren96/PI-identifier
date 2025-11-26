import cv2
import numpy as np
import matplotlib.pyplot as plt
import assisting_functions as af
import multiprocessing
import pandas as pd
from scipy.signal import find_peaks
import os
import re

make_plots_flag = False

# Loading Image and select ROI


#ID = ['149', '152', '153', '171']###### redo

#ID = ['131a']
#ID = ['145', '147', '149']
#ID = ['151', '152', '153', '154', '155', '156', '157', '158a']
#ID = ['158b', '159']
#ID = ['160', '161', '170a', '170b'] # ORANGE
#ID = ['163', '164', '165']
#ID = ['166', '167', '168', '169']
#ID = ['173', '174', '175', '176', '177', '178', '179']
#ID = ['180a', '180b', '181', '182', '183', '184', '185']
#ID = ['186', '187', '188', '189', '190', '191a', '191b']
#ID = ['192', '193', '194a', '195', '197', '198', '199a', '199b']
#ID = ['196', '200', '201']
#ID = ['202', '203', '204', '205']
#ID = ['206', '207', '209', '210', '211a', '211b']
#ID = ['212', '213', '214', '215', '216a', '216b']
#ID = ['217', '218', '221', '222', '224']
#ID = ['225', '226', '228', '229']
#ID = ['230', '232', '233', '234a', '234b']
#ID = ['235', '237', '238', '239']
#ID = ['231', '236']
#ID = ['240', '241a', '241b', '242', '243']
#ID = ['149', '152', '153', '171']
#ID = ['244', '246a', '246b', '247']
#ID = ['245', '248', '249', '250', '251']
import matplotlib.pyplot as plt
import numpy as np

# Create figure and axis
fig, ax = plt.subplots()

# Set limits
ax.set_xlim(0, 5500)
ax.set_ylim(0, 1.4)

# Set major and minor ticks
ax.set_xticks(np.arange(0, 5501, 500))   # Major ticks every 500
ax.set_xticks(np.arange(0, 5501, 100), minor=True)  # Minor ticks every 100
ax.set_yticks(np.arange(0, 1.41, 0.2))   # Major ticks every 0.2
ax.set_yticks(np.arange(0, 1.41, 0.05), minor=True)  # Minor ticks every 0.05

# Add grid with different styles
ax.grid(True, which="major", linewidth=0.8)  # Major grid
ax.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.6)  # Minor grid

# Labels
ax.set_xlabel(r"$r$ [m]")
ax.set_ylabel(r"$I$ [W]")
ax.set_title("Graph with Grid")

# Show plot
plt.show()





ID = [ '132a']


GW = pd.read_csv(fr'C:\Python_Projects\pythonProject\gw.csv')
id_int = [int(re.match(r'\d+', id).group()) for id in ID]

gw = GW[GW['id_int'].isin(id_int)]

fs = []
rectcrop = []
pxl_AMP = []
zero_line = []

df = pd.DataFrame(columns=gw.columns)
j=0

for id in ID:
    gw_row = gw[gw['id_int'] == id_int[j]].reset_index(drop=True)
    # Check if a matching row is found
    if not gw_row.empty:
        # Insert the first matching row into 'df' at index 'i'
        df.loc[j] = gw_row['gw'].iloc[0]  # Take only the first row from 'gw_row'
    else:
        print(f"No match found for id_int[{id}]")
    df.loc[j, 'id'] = id
    I = fr'C:\Python_Projects\DB\{id}.jpg'  # Use raw string for file path
    Y_SKIP = 400
    A_full = cv2.imread(I)
    A = A_full[Y_SKIP:, :]
    height, width, _ = A.shape
    print("\n*************\nID: ", id)
    ############################################
    # Zero line extraction
    img = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    img2 = img[0:-15, 149:-150]
    #
    # median_values = np.median(img2, axis=1) / 255
    # var_values = np.var(img2.astype(float), axis=1, ddof=0)
    # diff_values = np.diff(img2.astype(float), axis=1)
    # #sum_diff_values = np.sum(np.abs(diff_values), axis=1)
    # sc_var = 0.999 / (1 + var_values)
    # sc_median = 0.001 * median_values
    # score = sc_var + sc_median
    #
    # for i in range(len(score)):
    #     if median_values[i] < (140 / 255):
    #         score[i] = 0
    #
    # zero_line_id = np.argmax(score)
    #####################################################
    # User cropping
    # Selecting ROI
    # Get screen resolution (adjust this to match your screen)
    screen_res = (1320, 685)  # Replace with your screen resolution
    orig_height, orig_width = A.shape[:2]
    # Resize the image to fit the screen resolution without maintaining aspect ratio
    resized_image = cv2.resize(A, screen_res)
    # Calculate the resize ratios
    width_ratio = screen_res[0] / orig_width
    height_ratio = screen_res[1] / orig_height



    # Create a resizable window
    cv2.namedWindow(fr'Select ROI including zero-line, id= {id}', cv2.WINDOW_NORMAL)

    # Resize the window to fit the screen
    cv2.resizeWindow(fr'Select ROI including zero-line, id= {id}', screen_res[0], screen_res[1])

    # Add text to the resized image
    cv2.putText(resized_image, str(id), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    # Display the resized image
    cv2.imshow(fr'Select ROI including zero-line, id= {id}', resized_image)

    # Select ROI on the resized image
    resized_pixel_x1, resized_pixel_y1, resized_w, resized_h = cv2.selectROI(fr'Select ROI including zero-line, id= {id}', resized_image, False)

    # Destroy the window after ROI selection
    cv2.destroyAllWindows()

    # Now you can use the selected ROI coordinates (x1, y1, w, h) for further processing
    #print(f"Selected ROI: x={x1}, y={y1}, width={w}, height={h}")
    splice = [resized_pixel_x1, resized_pixel_y1, resized_w, resized_h]
    #splice_original = [splice[0]/width_ratio, splice[1]/height_ratio, splice[2]/height_ratio, splice[3]/width_ratio]
    img3 = resized_image[splice[1]: splice[1]+splice[3], splice[0]:splice[0]+ splice[2]]
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    median_values = np.median(img3, axis=1) / 255
    var_values = np.var(img3.astype(float), axis=1, ddof=0)
    diff_values = np.diff(img3.astype(float), axis=1)
    # sum_diff_values = np.sum(np.abs(diff_values), axis=1)
    sc_var = 0.999 / (1 + var_values)
    sc_median = 0.001 * median_values
    score = sc_var + sc_median

    for i in range(len(score)):
        if median_values[i] < (140 / 255):
            score[i] = 0

    zero_line_id_new = np.argmax(score)

    original_x, original_y = af.get_original_coordinates(resized_pixel_x1, resized_pixel_y1, width_ratio, height_ratio)
    # Calculate the original width and height of the ROI
    original_w = int(resized_w / width_ratio)
    original_h = int(resized_h / height_ratio)
    original_zero_line_id_new = int(zero_line_id_new / height_ratio)
    rectout = [Y_SKIP+original_y, Y_SKIP + original_zero_line_id_new+ original_y, original_x, original_x+original_w]

    I2 = A_full[rectout[0]: rectout[1], rectout[2]: rectout[3]]
    et1, et2, et3, = af.display_image_with_cross_cursor(I2, ' Verify rectcrop_im  click anywhere to continue')

    # Y-axis spacing calculation
    print('Please LEFT click the Y value on the "45", RIGHT click on the "60" line, or MIDDLE click for else : ')
    x, ref_line, event_type = af.display_image_with_cross_cursor(A, 'Please LEFT click the Y value on the "45", RIGHT click on the "60" line, or MIDDLE click for else : ')

    if event_type == 'left_click':  # the "45" line
        reference_AMP = 45
    elif event_type == 'right_click':
        reference_AMP = 60
    elif event_type == 'middle_click':
        reference_AMP = int(input('Please enter the Y line you clicked on: '))

    print(f'Entered:   ', reference_AMP)
    pxl_AMP_id = abs(ref_line - zero_line_id_new) / reference_AMP

    # fs calculation using time spacing
    # Ensure all previous figures are closed
    plt.close('all')
    boundary1, y, et = af.display_image_with_cross_cursor(A, 'Time spacing calculation, please click the FIRST pixel boundery : ')

    # Ensure all previous figures are closed
    plt.close('all')
    boundary2, y, et = af.display_image_with_cross_cursor(A, 'Time spacing calculation, please click the SECOND pixel boundery : ')
    # Round to the nearest 10's
    fs_id = round(abs(boundary1 - boundary2), -1)


    if id == ID[0]:
        rectcrop = [list(rectout)]
        zero_line = zero_line_id_new
        pxl_AMP = pxl_AMP_id
        fs = fs_id
        print(rectcrop)

    else:
        rectcrop = rectcrop + [list(rectout)]
        zero_line = np.vstack([zero_line, zero_line_id_new])
        pxl_AMP = np.vstack([pxl_AMP, pxl_AMP_id])
        fs = np.vstack([fs, fs_id])

    j+=1

df['fs'] = fs
df['pxl_AMP'] = pxl_AMP
df['initial_zero_line'] = zero_line
df['rectcrop'] = rectcrop

verify_column_order = ['id', 'gw', 'fs', 'pxl_AMP',	'initial_zero_line', 'rectcrop']
df = df[verify_column_order]

file_path = fr'C:\Python_Projects\pythonProject\stabs.csv'
if os.path.exists(file_path):
    # If the file exists, append to it
    try:
        df.to_csv(file_path, mode='a', index=False, header=False)
    except PermissionError as e:
        print("Permission Error:", e)
        # Handle the permission error here, such as by informing the user or taking corrective action
        input('Close file and press ENTER')
        df.to_csv(file_path, mode='a', index=False, header=False)

else:
    # If the file does not exist, create it and write the DataFrame to it
    try:
        df.to_csv(file_path, index=False, header=False)
    except PermissionError as e:
        print("Permission Error:", e)
        # Handle the permission error here, such as by informing the user or taking corrective action
        input('Close file and press ENTER\n')
        df.to_csv(file_path, index=False, header=False)


print('Finished')
input("Press Enter to exit...")

