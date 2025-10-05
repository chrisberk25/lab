import numpy as np
import matplotlib.pyplot as plt
from picarx import Picarx
from vilib import Vilib

N = 100 

def main():
    try:
        px = Picarx()

        Vilib.camera_start(vflip=False, hflip=False)
        Vilib.display(local=True, web=True)
        Vilib.show_fps()

        fig, ax = plt.subplots()
        map = np.ones((N, N)) 
        im = ax.imshow(map, cmap="gray_r", interpolation="nearest", origin="lower")
        plt.colorbar(im, ax=ax, label="Intensity")

        while True:
            map = np.ones((N, N))
            coords = []

            for i in range(90):
                angle = i - 45
                px.set_cam_pan_angle(angle)
                r = px.ultrasonic.read()

                x = int(np.cos(np.deg2rad(angle)) * r)
                y = int(np.sin(np.deg2rad(angle)) * r + N / 2)

                print("len:", r, "; angle:", angle)
                coords.append((x, y))

                if 0 <= x < N and 0 <= y < N:
                    map[x, y] = 0  # black obstacle

            # Update the existing image data
            im.set_data(map)
            plt.draw()
            plt.pause(0.01)

            # Optional: save current scan
            plt.savefig("my_plot.png")

    finally:
        px.forward(0)


if __name__ == "__main__":
    main()
