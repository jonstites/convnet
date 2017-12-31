from collections import Counter
import glob
import hashlib
import imageio
import itertools
import numpy as np
import os


class ImageDataset:

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self.images.shape[0]

    def train_validate_split(self, train_fraction=0.9):
        np.random.seed(0)
        num_examples = len(self.images)
        perm = np.random.permutation(self._num_examples)
        split_pos = int(num_examples * train_fraction)
        train_perm = perm[:split_pos]
        train_dataset = ImageDataset(self.images[train_perm], self.labels[train_perm])
        val_perm = perm[split_pos:]
        val_dataset = ImageDataset(self.images[val_perm], self.labels[val_perm])
        return train_dataset, val_dataset
                                     
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0:
            self._shuffle_data()
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            images_first_part = self.images[start:]
            labels_first_part = self.labels[start:]
            self._shuffle_data()

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_last_part = self.images[start:end]
            labels_last_part = self.labels[start:end]
            batch_images = np.concatenate((images_first_part, images_last_part), axis=0)
            batch_labels = np.concatenate((labels_first_part, labels_last_part), axis=0)
            return batch_images, batch_labels
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.images[start:end], self.labels[start:end]

    def _shuffle_data(self):
        perm = np.random.permutation(self._num_examples)
        self.images = self.images[perm]
        self.labels = self.labels[perm]

    
class NotMNISTDataset:

    def __init__(self, train_dir, test_dir):
        print("loading data")
        self.train = self.load_data(train_dir)
        self.test = self.load_data(test_dir)
        print("removing duplicates")
        self._find_and_remove_duplicates()
        print("splitting")
        self.train, self.validate = self.train.train_validate_split()
        self.whiten_images()
        
    def load_data(self, data_dir):
        image_filenames, char_labels = self._get_image_filenames_and_labels(data_dir)
        labels = []
        images = []
        for i, image_file in enumerate(image_filenames):
            try:
                image = imageio.imread(image_file).astype(np.uint8)
                images.append(image.reshape([28, 28, 1]))
                labels.append(ord(char_labels[i]) - ord('A'))
            except ValueError:
                print("can't open file: ", image_file)

        return ImageDataset(np.asarray(images), np.asarray(labels))

    def _get_image_filenames_and_labels(self, data_dir):
        label_regex = os.path.join(data_dir, "*")
        image_filenames = []
        labels = []
        for label_dir in glob.glob(label_regex):
            label = os.path.basename(label_dir)
            image_regex = os.path.join(label_dir, "*.png")
            for image_file in glob.glob(image_regex):
                image_filenames.append(image_file)
                labels.append(label)
        return image_filenames, labels
    
    def _find_and_remove_duplicates(self):
        duplicates = self._find_duplicates()
        self.train = self._remove_duplicates(self.train, duplicates)
        self.test = self._remove_duplicates(self.test, duplicates)

    def _find_duplicates(self):
        hashes = Counter()
        for image in itertools.chain(self.train.images, self.test.images):
            hash_value = self._image_hash(image)
            hashes[hash_value] += 1
        duplicates = set([h for h, count in hashes.items() if count > 1])
        print("found ", len(duplicates), " duplicates.")
        return duplicates

    def _image_hash(self, image):
        return hashlib.sha256(image).digest()
    
    def _remove_duplicates(self, dataset, duplicates):
        images = []
        labels = []
        
        for image, label in zip(dataset.images, dataset.labels):
            if not self._image_hash(image) in duplicates:
                images.append(image)
                labels.append(label)
        return ImageDataset(np.asarray(images), np.asarray(labels))

    def whiten_images(self):
        mean = np.mean(self.train.images)
        std = np.std(self.train.images)

        self.train.images = (self.train.images - mean) / std
        self.validate.images = (self.validate.images - mean) / std
        self.test.images = (self.test.images - mean) / std        
