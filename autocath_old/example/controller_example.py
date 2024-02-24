from time import perf_counter, sleep
import os
import numpy as np
import autocath


def create_random_data(controller):
    n_tracking_points = np.random.randint(1, 7)
    tracking_shape = (n_tracking_points, 2)
    tracking = np.random.random_sample(tracking_shape) * 2 - 1
    target_coords = np.random.random_sample((1, 2)) * 2 - 1
    device_lengths_inserted = np.random.random_sample(controller.n_devices) * 500
    device_lengths_inserted = device_lengths_inserted.tolist()
    return tracking, target_coords, device_lengths_inserted


image_frequency = 7.5

dir = os.path.dirname(os.path.realpath(__file__))
cp = os.path.join(dir, "checkpoint4800089")
config = os.path.join(dir, "config.yml")
controller = autocath.RViMController1(
    checkpoint=cp,
    config=config,
    tracking_high=[1, 1],
    tracking_low=[-1, -1],
)


for _ in range(5):
    # controller needs to be reset at the beginning of the navigation:
    tracking, target_coords, device_lengths_inserted = create_random_data(controller)
    controller.reset(
        tracking,
        target_coords,
        device_lengths_inserted,
        [1, 1],
        [-1, -1],
    )
    for _ in range(150):
        t_start = perf_counter()
        tracking, target_coords, device_lengths_inserted = create_random_data(
            controller
        )

        # this is the important step:
        action = controller.step(tracking, target_coords, device_lengths_inserted)
        print(action)

        t_duration = perf_counter() - t_start
        t_duration_d = 1 / image_frequency
        if t_duration_d > t_duration:
            sleep(t_duration_d - t_duration)
