
# def save_images_from_event(fn, tag, output_dir="./"):
#     assert os.path.isdir(output_dir)
    
#     image_str = tf.placeholder(tf.string)

#     image_str = tf.placeholder(tf.string)
#     im_tf = tf.image.decode_image(image_str)

#     sess = tf.InteractiveSession()
#     with sess.as_default():
#         count = 0
#         for e in tf.train.summary_iterator(fn):
#             for v in e.summary.value:
#                 if v.tag == tag:
#                     im = im_tf.eval({image_str: v.image.encoded_image_string})
#                     output_fn = os.path.realpath(
#                         "{}/image_{:05d}.png".format(output_dir, count)
#                     )
#                     print("Saving '{}'".format(output_fn))
#                     scipy.misc.imsave(output_fn, im)
#                     count += 1

from collections import defaultdict, namedtuple
from typing import List
import tensorflow as tf


TensorBoardImage = namedtuple("TensorBoardImage", ["topic", "image", "cnt"])


def extract_images_from_event(event_filename: str, image_tags: List[str]):
    topic_counter = defaultdict(lambda: 0)

    serialized_examples = tf.data.TFRecordDataset(event_filename)
    for serialized_example in serialized_examples:
        event = event_pb2.Event.FromString(serialized_example.numpy())
        for v in event.summary.value:
            if v.tag in image_tags:

                if v.HasField('tensor'):  # event for images using tensor field
                    s = v.tensor.string_val[2]  # first elements are W and H

                    tf_img = tf.image.decode_image(s)  # [H, W, C]
                    np_img = tf_img.numpy()

                    topic_counter[v.tag] += 1

                    cnt = topic_counter[v.tag]
                    tbi = TensorBoardImage(topic=v.tag, image=np_img, cnt=cnt)

                    yield tbi


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract images from a tensorboard event file"
    )
    parser.add_argument(
        "event_file",
        default="../../data/lightning_logs/version_112/events.out.tfevents.1713704034.DESKTOP-Andries.353620.0",
    )
    parser.add_argument(
        "tag", default="Reconstructions channel 1", help="The tag to extract images for"
    )
    parser.add_argument(
        "--output-dir", default="./", help="The directory to save images to"
    )
    args = parser.parse_args()

    save_images_from_event(args.event_file, args.tag, args.output_dir)

    # /home/andries/projects/mai-thesis/data/lightning_logs/version_112/events.out.tfevents.1713704034.DESKTOP-Andries.353620.0
