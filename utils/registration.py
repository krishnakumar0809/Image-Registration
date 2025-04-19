import SimpleITK as sitk
from PIL import Image
import numpy as np

def register_images(fixed_path, moving_path):
    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetInterpolator(sitk.sitkLinear)

    transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler2DTransform())
    registration_method.SetInitialTransform(transform, inPlace=False)

    final_transform = registration_method.Execute(fixed, moving)
    resampled = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())

    array = sitk.GetArrayFromImage(resampled)
    array = (array - array.min()) / (array.max() - array.min()) * 255
    return Image.fromarray(array.astype(np.uint8))
