import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt



####################################################################################
########################### Phase 1, Data Collection ###############################
####################################################################################

# Global Variables
csv_file = "rock_data.csv"
h1, s1, v1, h2, s2, v2, h3, s3, v3 = 0, 0, 0, 0, 0, 0, 0, 0, 0

def plot_histogram(h, s, v, name, loc):
    plt.clf()
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(h, color='r')
    plt.title('Hue')
    plt.subplot(132)
    plt.plot(s, color='g')
    plt.title('Saturation')
    plt.subplot(133)
    plt.plot(v, color='b')
    plt.title('Value')
    
    # Save the figure
    plt.savefig(f"{loc}/{name}.png")
    plt.close()

def plot_histogram3(h1, s1, v1, h2, s2, v2, h3, s3, v3):
    plt.figure(figsize=(15, 15))
    
    plt.subplot(331)
    plt.plot(h1, color='r')
    plt.title('Hue 1')
    
    plt.subplot(332)
    plt.plot(s1, color='g')
    plt.title('Saturation 1')
    
    plt.subplot(333)
    plt.plot(v1, color='b')
    plt.title('Value 1')
    
    plt.subplot(334)
    plt.plot(h2, color='r')
    plt.title('Hue 2')
    
    plt.subplot(335)
    plt.plot(s2, color='g')
    plt.title('Saturation 2')
    
    plt.subplot(336)
    plt.plot(v2, color='b')
    plt.title('Value 2')
    
    plt.subplot(337)
    plt.plot(h3, color='r')
    plt.title('Hue 3')
    
    plt.subplot(338)
    plt.plot(s3, color='g')
    plt.title('Saturation 3')
    
    plt.subplot(339)
    plt.plot(v3, color='b')
    plt.title('Value 3')
    
    plt.tight_layout()
    plt.show()

def create_csv():
    global h1, s1, v1, h2, s2, v2, h3, s3, v3
    class_folder = [[1, 14], [2, 20], [3, 13]]

    for i in range(3):
        for j in range(class_folder[i][1]):
            image = cv2.imread(f"../../dataset/class_{i+1}/C{i+1}_{j+1}.jpg")
            hsv_colors = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Histograms
            hist_hue = cv2.calcHist([hsv_colors], [0], None, [180], [0, 180])
            hist_sat = cv2.calcHist([hsv_colors], [1], None, [256], [0, 256])
            hist_val = cv2.calcHist([hsv_colors], [2], None, [256], [0, 256])

            name = f"C{i+1}_{j+1}_Histogram"
            loc = f"../../dataset/class_{i+1}"

            plot_histogram(hist_hue, hist_sat, hist_val, name, loc)

            # done = False

            # if i == 0 and j == 1:
            #     h1 = hist_hue
            #     s1 = hist_sat
            #     v1 = hist_val
            # elif i == 1 and j == 1:
            #     h2 = hist_hue
            #     s2 = hist_sat
            #     v2 = hist_val
            # elif i == 2 and j == 1:
            #     h3 = hist_hue
            #     s3 = hist_sat
            #     v3 = hist_val
            #     done = True

                

            

            # if done:
            #     plot_histogram3(h1, s1, v1, h2, s2, v2, h3, s3, v3)
            #     done = False


            hist_hue_flat = hist_hue.flatten()
            hist_sat_flat = hist_sat.flatten()
            hist_val_flat = hist_val.flatten()

            hist_combined = np.concatenate([hist_hue_flat, hist_sat_flat, hist_val_flat])
            hist_combined = np.append(hist_combined, i+1)

            with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(hist_combined)


####################################################################################
########################### Phase 2, Model Train ###################################
####################################################################################        

if __name__ == "__main__":
    # create_csv()

    print("Model Training Started...")

    print("Model Training Completed...")

    






















