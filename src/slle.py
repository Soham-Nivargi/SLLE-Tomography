from skimage.transform import radon, iradon, rotate
from sklearn.manifold import locally_linear_embedding
from scipy.sparse.linalg import eigsh
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigs
from scipy.linalg import orthogonal_procrustes
from sklearn.preprocessing import normalize
import numpy as np
import os, argparse, cv2
import matplotlib.pyplot as plt

NOISE_LEVEL = 0.01
np.random.seed(42)

def get_rotate_image(img1, img2):
    min_value = 1000
    rotated_img = img2
    for i in range(-20, 20, 1):
        rotated_ = rotate(img2, i)
        mse = np.mean((img1 - rotated_) ** 2)
        if mse < min_value:
            rotated_img = rotated_
            min_value = mse
    return rotated_img

def compute_rrmse_snr(img1, img2):
    img2 = get_rotate_image(img1, img2)
    mse = np.mean((img1 - img2) ** 2)
    rrmse = np.sqrt(mse) / np.sqrt(np.mean(img1 ** 2))
    snr = 10 * np.log10(np.sum(img1 ** 2) / np.sum((img1 - img2) ** 2))
    return rrmse, snr

def generate_sinogram(image, number_angles, theta=None):
    angles = np.random.uniform(0, 180, number_angles)
    angles.sort()
    sinograms = radon(image, theta=angles)
    noise = np.random.normal(0, NOISE_LEVEL * np.max(sinograms), sinograms.shape)
    sinograms += noise
    return sinograms, angles

def estimate_angles(sinograms, n_neighbors, n_components = 2):
    fft_sinograms = np.abs(np.fft.fft(sinograms, axis=1))
    n_samples = sinograms.shape[0]
    embedding, err = locally_linear_embedding(
            fft_sinograms, 
            n_neighbors=n_neighbors,
            n_components=n_components + 1,
            method='standard',
            random_state=42
    )

    embedding = embedding[:,1:]
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    angles_estimated = np.degrees(np.arctan2(embedding[:, 1], embedding[:, 0]))
    angles_estimated = (angles_estimated + 360) % 180
    angles_estimated.sort()
    return angles_estimated

def main(file_name):
    base_name = os.path.basename(file_name)
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    rrmse_list, snr_list = [], []
    angle_list = np.arange(30, 160, 40)
    for i in angle_list:
        sinograms, true_angles = generate_sinogram(image, i)
        angles = estimate_angles(sinograms.T.copy(), 12)
        idx = np.argsort(angles);
        sorted_projections = sinograms[:, idx];
        recon_angles = np.linspace(0, 180, i);
        reconstruction = iradon(sorted_projections, theta=recon_angles, filter_name="ramp", circle=True)
        rrmse, snr = compute_rrmse_snr(image, reconstruction)
        rrmse_list.append(rrmse)
        snr_list.append(snr)
        plt.imsave(f"results/reconstructed_{i}_{base_name}", reconstruction, cmap='gray')

    plt.plot(angle_list, rrmse_list, label='RRMSE', marker='o')
    plt.xlabel('Number of angles')
    plt.ylabel('RRMSE')
    plt.savefig(f"results/rrmse_{base_name}.png")
    plt.close()

    plt.plot(angle_list, snr_list, label='SNR', marker='o')
    plt.xlabel('Number of angles')
    plt.ylabel('SNR')
    plt.savefig(f"results/snr_{base_name}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sinograms from a phantom image.")
    parser.add_argument("--file_name", type=str, help="Path to the image file.", required=True)
    args = parser.parse_args()
    main(args.file_name)
