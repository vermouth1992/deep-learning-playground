from tqdm.auto import tqdm
import tensorflow as tf

data = tf.data.Dataset.range(10000).batch(32)

def main():
    for i in tqdm(data, total=10000 / 32):
        pass

if __name__ == '__main__':
    main()

