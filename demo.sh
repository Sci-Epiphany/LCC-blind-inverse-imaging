python posterior_sample.py \
+data=test-ffhq    \
+kernel_data=gaussian64   \
+model=ffhq256ddpm    \
+kernel_model=kernel64ddpm   \
+task=blind_gaussian_blur  \
+sampler=edm_daps   \
task_group=pixel    \
save_dir=results    \
num_runs=1     \
sampler.diffusion_scheduler_config.num_steps=5     \
sampler.annealing_scheduler_config.num_steps=200    \
sampler.blind=True   \
sampler.nonlinear=False  \
batch_size=1    \
data.start_id=0    \
data.end_id=1    \
name=gaussian_blur_demo  \
gpu=1

#python posterior_sample.py \
#+data=test-ffhq   \
#+kernel_data=nonlinear  \
#+model=ffhq256ddpm   \
#+task=blind_nonlinear_blur  \
#+sampler=edm_daps  \
#task_group=pixel  \
#save_dir=results  \
#num_runs=1  \
#sampler.diffusion_scheduler_config.num_steps=5  \
#sampler.annealing_scheduler_config.num_steps=200   \
#batch_size=32     \
#data.start_id=0    \
#data.end_id=32   \
#name=nonlinear_blur_demo  \
#gpu=1

#python posterior_sample.py \
#+data=test-imagenet   \
#+kernel_data=nonlinear  \
#+model=imagenet256ddpm   \
#+task=blind_nonlinear_blur  \
#+sampler=edm_daps  \
#task_group=pixel  \
#save_dir=results  \
#num_runs=4  \
#sampler.diffusion_scheduler_config.num_steps=5  \
#sampler.annealing_scheduler_config.num_steps=200   \
#batch_size=32     \
#data.start_id=0    \
#data.end_id=16   \
#name=nonlinear_blur_test  \
#gpu=1
