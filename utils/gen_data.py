from argparse import ArgumentParser
from astropy.modeling.functional_models import AiryDisk2D, Gaussian2D
from george import GP
from george import kernels
import json
import numpy as np
import os
from skimage.io import imsave
from tqdm import tqdm


def create_psf_model(param_file):
    with open(param_file, "r") as json_file:
        param = json.load(json_file)

    model = (
        AiryDisk2D(**param["AiryDisk2D"], name="core")
      + Gaussian2D(**param["Gaussian2D"], name="halo"))

    # check that core is centered
    assert np.isclose(model["core"].x_0, 0)
    assert np.isclose(model["core"].y_0, 0)

    # check that halo and core are concentric
    assert np.isclose(model["halo"].x_mean, model["core"].x_0)
    assert np.isclose(model["halo"].y_mean, model["core"].y_0)

    # check that halo is symmetric
    assert np.isclose(model["halo"].x_stddev, model["halo"].y_stddev)

    # check that the model is correctly normalized
    assert np.isclose(model(0, 0), 1)

    return model

def create_peak_model():
    return lambda x, y, sigma: np.exp(-(x**2+y**2)/(2*sigma**2))


def create_grid(image_size, image_box):
    x_min, x_max = image_box[0]
    y_min, y_max = image_box[1]

    dx = (x_max - x_min)/image_size[0]
    dy = (y_max - y_min)/image_size[1]
    assert np.isclose(dx, dy)

    x = np.linspace(x_min + 0.5*dx, x_max - 0.5*dx, image_size[0])
    y = np.linspace(y_min + 0.5*dy, y_max - 0.5*dy, image_size[1])

    return np.meshgrid(x, y)

def normalize(image, min_value, max_value):
    return (image - min_value)/(max_value - min_value)    


def create_background(grid, scale_length, variance):
    x, y = grid

    kernel = kernels.Matern32Kernel(scale_length, ndim=2)
    gp = GP(variance*kernel)

    samples = gp.sample(np.transpose(np.vstack([x.ravel(), y.ravel()])))
    return np.transpose(samples).reshape(*x.shape)

def create_foreground(grid, image_box, n_total, psf_model, scale, alpha, flux_min, core_frac=None):
    x, y = grid
    x_min, x_max = image_box[0]
    y_min, y_max = image_box[1]

    psf = psf_model.copy()
    peak = create_peak_model()

    if core_frac is not None:
        psf["core"].amplitude = core_frac
        psf["halo"].amplitude = 1 - core_frac

    assert np.isclose(psf(0, 0), 1)
    assert np.isclose(peak(0, 0, 1), 1)
    
    image = np.zeros(x.shape)
    sharp_image = np.zeros_like(image)
    for n in range(n_total):
        x0 = np.random.uniform(x_min, x_max)
        y0 = np.random.uniform(y_min, y_max)
        flux = flux_min * (np.random.pareto(alpha) + 1)
        image += flux * psf((x-x0)*scale, (y-y0)*scale)
        sharp_image += flux * peak((x-x0)*scale, (y-y0)*scale, 3/np.sqrt(8*np.log(2)))

    return image, sharp_image

def create_noise(grid, sigma):
    x, y = grid
    return np.random.normal(0, sigma, size=x.shape)


def create_pairs(image_size, psf_model, target):
    x, y = create_grid((image_size, image_size), [[-1, 1]]*2)
    
    ## generate background image
    
    scale_length = np.random.uniform(2, 4)

    variance = 10**np.random.uniform(-3, 0)
    
    background_image = create_background((x, y),
        scale_length=scale_length,
        variance=variance)
    
    ## generate foreground image
    
    n_total = np.random.randint(1, 16)

    true_scale = image_size/2
    min_scale = 0.8*true_scale
    max_scale = 1.2*true_scale
    scale = 10**np.random.uniform(np.log10(min_scale), np.log10(max_scale))

    true_frac = psf_model["core"].amplitude
    min_frac = np.max([0, 0.8*true_frac])
    max_frac = np.min([1, 1.2*true_frac])
    core_frac = np.random.uniform(min_frac, max_frac)

    alpha = 0.7

    flux_min = 0.1

    foreground_image, sharp_image = create_foreground((x, y),
        image_box=[[-2, 2]]*2, # account for sources outside of image box
        n_total=4*n_total, # account for the larger area of the sampling box
        psf_model=psf_model,
        scale=scale,
        core_frac=core_frac,
        alpha=alpha,
        flux_min=flux_min)

    ## generate noise image
    
    sigma = 10**(np.random.uniform(-3, -2))

    noise_image = create_noise((x, y), sigma=sigma)

    ## define input and target images

    image = background_image + foreground_image + noise_image

    if target == "background":
        target = background_image # do foreground subtraction
        target = normalize(target, np.min(image), np.max(image))
    elif target == "foreground":
        target = sharp_image # do background subtraction
        target = normalize(target, 0, np.max(foreground_image))

    image = normalize(image, np.min(image), np.max(image))

    return image, target

def export_data(output_dir, n_samples, seed=None, **kwargs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  

    if seed is not None:
        np.random.seed(seed)

    for i in tqdm(range(n_samples)):
        while True:
            try:
                image, target = create_pairs(**kwargs)
                imsave(os.path.join(output_dir, "%d.png" % (i+1)),
                       np.hstack((image, target)))
                break
            except:
                pass


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--param_file", required=True)
    parser.add_argument("--target", default="background")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_train", type=int, required=True)
    parser.add_argument("--n_val", type=int)
    parser.add_argument("--seed_train", type=int)
    parser.add_argument("--seed_val", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    opts = parse_arguments()

    psf_model = create_psf_model(opts.param_file)

    export_data(os.path.join(opts.output_dir, "train"),
        n_samples=opts.n_train,
        seed=opts.seed_train,
        image_size=opts.image_size,
        psf_model=psf_model,
        target=opts.target)
    
    if opts.n_val is not None:
        export_data(os.path.join(opts.output_dir, "val"),
            n_samples=opts.n_val,
            seed=opts.seed_val,
            image_size=opts.image_size,
            psf_model=psf_model,
            target=opts.target)
