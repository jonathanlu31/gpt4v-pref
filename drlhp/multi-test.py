from multiprocessing import Process
import gymnasium as gym


def main():
    def child():
        env = gym.make("Swimmer-v4", render_mode="rgb_array")
        env.reset()
        print(env.render())

    p = Process(target=child, daemon=True)
    p.start()
    p.join()


main()
