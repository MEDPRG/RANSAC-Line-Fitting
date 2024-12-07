
# RANSAC Line Fitting with Python and OpenCV

This repository contains a Python script for implementing the RANSAC (Random Sample Consensus) algorithm to fit a line to a dataset of 2D points. It visualizes the inliers and the best-fitted line on a synthetic dataset containing inliers and outliers.

## Features

- **Synthetic Data Generation**:
  - Generates a dataset of 2D points with a specified ratio of inliers and outliers.

- **Line Fitting**:
  - Implements line fitting using least squares and RANSAC to handle noisy datasets with outliers.

- **Interactive Visualization**:
  - Displays the generated points, the true line, and the fitted line using OpenCV.

## Requirements

To run the script, ensure you have the following dependencies installed:

- Python 3.x
- OpenCV (`cv2`)
- NumPy

Install the required packages using pip:

```bash
pip install opencv-python numpy
```

## Usage

1. **Run the Script**:
   - Execute the script to generate a synthetic dataset, fit a line using RANSAC, and visualize the result:
     ```bash
     python ransac_line_fitting.py
     ```

2. **View Results**:
   - The script displays:
     - **Generated Points**: Inliers and outliers in white.
     - **True Line**: Represented in blue.
     - **Fitted Line**: Represented in green, computed using RANSAC.

## Input Data

The script generates a synthetic dataset with inliers aligned along a line and outliers scattered randomly. No external input is required.

## Output

The output is displayed in a window, visualizing the points and the lines. The RANSAC algorithm identifies the best-fit line while rejecting outliers.

## Example

Here’s an example of the script in action:

1. **Generated Points**:
   - A dataset with 50% inliers and 50% outliers.
    ![image](https://github.com/user-attachments/assets/6dcdc718-364b-483c-9457-5caf293f05f3)

2. **Visualization**:
   - The true line (blue) and the RANSAC-fitted line (green).
   ![image](https://github.com/user-attachments/assets/4456bc64-543f-4f26-a04c-444597d38736)


---

## Author

**MEDPRG**  
[GitHub Profile](https://github.com/MEDPRG)
