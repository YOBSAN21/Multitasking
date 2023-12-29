import Augmentor


dir = 'C:/Users/PC/Desktop/Work_on_This_Lady/Lari/Larisa/Data_2/Val/Normal'

p = Augmentor.Pipeline(dir)
p.rotate(probability=1.0, max_left_rotation=5, max_right_rotation=10)
# p.zoom(probability=0.2, min_factor=1.1, max_factor=1.2)
p.skew(probability=0.2)
p.skew_left_right(probability=0.3)
p.skew_top_bottom(probability=0.3)
p.shear(probability=0.2, max_shear_left=2, max_shear_right=2)
p.crop_random(probability=0.5, percentage_area=0.8)
p.flip_random(probability=0.2)
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
p.flip_left_right(probability=0.4)
p.flip_top_bottom(probability=0.8)
p.rotate90(probability=0.1)
# p.rotate270(probability=0.5)

num_of_samples = 343
p.sample(num_of_samples)

