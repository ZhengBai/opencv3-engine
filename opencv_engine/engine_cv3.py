#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the MIT license:
# http://www.opensource.org/licenses/mit-license
# Copyright (c) 2016 fanhero.com christian@fanhero.com

import cv2
import numpy as np
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile
import os
from colour import Color
from thumbor.engines import BaseEngine
from pexif import JpegFile, ExifSegment
from thumbor.utils import logger, EXTENSION

try:
    from thumbor.ext.filters import _composite
    FILTERS_AVAILABLE = True
except ImportError:
    FILTERS_AVAILABLE = False

FORMATS = {
    '.heif': 'JPEG',
    '.tif': 'PNG',
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.gif': 'GIF',
    '.png': 'PNG',
    '.webp': 'WEBP'
}

class Gif2WebpError(RuntimeError):
    pass

class Engine(BaseEngine):
    @property
    def image_depth(self):
        if self.image is None:
            return np.uint8
        return self.image.dtype

    @property
    def image_channels(self):
        if self.image is None:
            return 3
        # if the image is grayscale
        try:
            return self.image.shape[2]
        except IndexError:
            return 1

    @classmethod
    def parse_hex_color(cls, color):
        try:
            color = Color(color).get_rgb()
            return tuple(c * 255 for c in reversed(color))
        except Exception:
            return None

    def gen_image(self, size, color_value):
        if color_value == 'transparent':
            color = (255, 255, 255, 255)
            img = np.zeros((size[1], size[0], 4), self.image_depth)
        else:
            img = np.zeros((size[1], size[0], self.image_channels), self.image_depth)
            color = self.parse_hex_color(color_value)
            if not color:
                raise ValueError('Color %s is not valid.' % color_value)
        img[:] = color
        return img

    def create_image(self, buffer):
        # FIXME: opencv doesn't support gifs, even worse, the library
        # segfaults when trying to decoding a gif. An exception is a
        # less drastic measure.
        try:
            if FORMATS[self.extension] == 'GIF':
                raise ValueError("opencv doesn't support gifs")
        except KeyError:
            pass

        img = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_UNCHANGED)
        if (FORMATS[self.extension] == 'PNG' and img.dtype == np.uint16 and len(img.shape) > 2 and img.shape[2] == 4) or \
                (FORMATS[self.extension] == 'JPEG' and img.dtype == np.uint8 and len(img.shape) > 2 and img.shape[2] == 4):
            img = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_ANYCOLOR)

        # cv2.imwrite("/Users/wangbaifeng/Downloads/testopencv.png", img)
        # img_png = cv2.imread("/Users/wangbaifeng/Downloads/testopencv.png", cv2.IMREAD_UNCHANGED)
        # png_options = [cv2.IMWRITE_JPEG_QUALITY, 80]
        # success, buf = cv2.imencode(".jpg", img_png, png_options or [])
        # cv2.imwrite("/Users/wangbaifeng/Downloads/testopencv.jpg", img_png)

        # imagefiledata = cv2.cv.CreateMatHeader(1, len(buffer), cv2.cv.CV_8UC1)
        # cv2.cv.SetData(imagefiledata, buffer, len(buffer))
        # img = cv2.cv.DecodeImageM(imagefiledata, cv2.cv.CV_LOAD_IMAGE_UNCHANGED)

        if FORMATS[self.extension] == 'JPEG':
            self.exif = None
            try:
                info = JpegFile.fromString(buffer).get_exif()
                if info:
                    self.exif = info.data
                    self.exif_marker = info.marker
            except Exception:
                pass
        return img

    @property
    def size(self):
        return self.image.shape[1], self.image.shape[0]

    def normalize(self):
        pass

    def resize(self, width, height):
        # r = height / self.size[1]
        # width = int(self.size[0] * r)
        dim = (int(round(width, 0)), int(round(height, 0)))
        self.command.extend(["-resize", str(round(width, 0)), str(round(height, 0))])
        self.image = cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA)

    def crop(self, left, top, right, bottom):
        width = right - left
        if right > self.size[0]:
            width = self.size[0] - left
        height = bottom - top
        if bottom > self.size[1]:
            height = self.size[1] - top
        self.command.extend(["-crop", str(left), str(top), str(width), str(height)])
        self.image = self.image[top: bottom, left: right]

    def rotate(self, degrees):
        # see http://stackoverflow.com/a/23990392
        if degrees == 90:
            self.image = cv2.transpose(self.image)
            cv2.flip(self.image, 0, self.image)
        elif degrees == 180:
            cv2.flip(self.image, -1, self.image)
        elif degrees == 270:
            self.image = cv2.transpose(self.image)
            cv2.flip(self.image, 1, self.image)
        else:
            # see http://stackoverflow.com/a/37347070
            # one pixel glitch seems to happen with 90/180/270
            # degrees pictures in this algorithm if you check
            # the typical github.com/recurser/exif-orientation-examples
            # but the above transpose/flip algorithm is working fine
            # for those cases already
            width, height = self.size
            image_center = (width / 2, height / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, degrees, 1.0)

            abs_cos = abs(rot_mat[0, 0])
            abs_sin = abs(rot_mat[0, 1])
            bound_w = int((height * abs_sin) + (width * abs_cos))
            bound_h = int((height * abs_cos) + (width * abs_sin))

            rot_mat[0, 2] += ((bound_w / 2) - image_center[0])
            rot_mat[1, 2] += ((bound_h / 2) - image_center[1])

            self.image = cv2.warpAffine(self.image, rot_mat, (bound_w, bound_h))

    def flip_vertically(self):
        self.image = np.flipud(self.image)

    def flip_horizontally(self):
        self.image = np.fliplr(self.image)

    def read(self, extension=None, quality=None):
        if quality is None:
            quality = self.context.config.QUALITY

        options = None
        extension = extension or self.extension
        try:
            if FORMATS[extension] == 'JPEG':
                options = [cv2.IMWRITE_JPEG_QUALITY, quality]
        except KeyError:
            # default is JPEG so
            options = [cv2.IMWRITE_JPEG_QUALITY, quality]

        try:
            if FORMATS[extension] == 'WEBP':
                options = [cv2.IMWRITE_WEBP_QUALITY, quality]
        except KeyError:
            options = [cv2.IMWRITE_JPEG_QUALITY, quality]

        # png with alpha convert jpg use white background
        if FORMATS[self.extension] == 'PNG' and FORMATS[extension] == 'JPEG' and self.image_channels == 4 and self.image.dtype == np.uint8:
            alpha_channel = self.image[:, :, 3]
            rgb_channels = self.image[:, :, :3]

            # White Background Image
            white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

            # Alpha factor
            alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
            alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

            # Transparent Image Rendered on White Background
            base = rgb_channels.astype(np.float32) * alpha_factor
            white = white_background_image.astype(np.float32) * (1 - alpha_factor)
            final_image = base + white
            success, buf = cv2.imencode(extension, final_image, options or [])
            return buf.tostring()

        # if FORMATS[self.extension] == 'PNG' and FORMATS[extension] == 'WEBP':
        #     png_file = NamedTemporaryFile(suffix='.png', delete=False)
        #     png_file.write(self.buffer)
        #     png_file.close()
        #
        #     output_suffix = '.webp'
        #     result_file = NamedTemporaryFile(suffix=output_suffix, delete=False)
        #     result_file.close()
        #
        #     logger.debug('convert {0} to {1}'.format(png_file.name, result_file.name))
        #     try:
        #         self.command.extend([
        #             '-q', str(quality),
        #             png_file.name,
        #             '-o', result_file.name
        #         ])
        #         png_2_webp_process = Popen(self.command, stdout=PIPE, stdin=PIPE, stderr=PIPE)
        #         png_2_webp_process.communicate()
        #         if png_2_webp_process.returncode != 0:
        #             raise Gif2WebpError(
        #                 'png2webp command returned errorlevel {0} for command "{1}"'.format(
        #                     png_2_webp_process.returncode, ' '.join(
        #                         self.command +
        #                         [self.context.request.url]
        #                     )
        #                 )
        #             )
        #         with open(result_file.name, 'r') as f:
        #             return f.read()
        #     finally:
        #         os.unlink(png_file.name)
        #         os.unlink(result_file.name)

        success, buf = cv2.imencode(extension, self.image, options or [])
        data = buf.tostring()

        if FORMATS[extension] == 'JPEG' and self.context.config.PRESERVE_EXIF_INFO:
            if hasattr(self, 'exif') and self.exif:
                img = JpegFile.fromString(data)
                img._segments.insert(0, ExifSegment(self.exif_marker, None, self.exif, 'rw'))
                data = img.writeString()

        return data

    def load(self, buffer, extension):
        self.extension = extension

        if extension is None:
            mime = self.get_mimetype(buffer)
            self.extension = EXTENSION.get(mime, '.jpg')

        if self.extension == '.tif':  # Pillow does not support 16bit per channel TIFF images
            buffer = self.convert_tif_to_png(buffer)

        if self.extension == '.heif':
            buffer = self.convert_heif_to_jpeg(buffer)

        super(Engine, self).load(buffer, self.extension)

    def convert_tif_to_png(self, buffer):
        if not cv2:
            msg = """[OpenCV Engine] convert_tif_to_png failed: opencv not imported"""
            logger.error(msg)
            return buffer

        img = cv2.imdecode(np.fromstring(buffer, dtype='uint16'), -1)
        buffer = cv2.imencode('.png', img)[1].tostring()
        mime = self.get_mimetype(buffer)
        self.extension = EXTENSION.get(mime, '.jpg')
        return buffer

    def convert_heif_to_jpeg(self, buffer):
        heif_file = NamedTemporaryFile(suffix='.heif', delete=False)
        heif_file.write(self.buffer)
        heif_file.close()

        output_suffix = '.jpg'
        result_file = NamedTemporaryFile(suffix=output_suffix, delete=False)
        try:
            logger.debug('convert {0} to {1}'.format(heif_file.name, result_file.name))
            result_file.close()
            command = [
                self.context.config.HEIF2JPEG_PATH,
                heif_file.name,
                result_file.name
            ]
            heif_2_jpg_process = Popen(command, stdout=PIPE, stdin=PIPE, stderr=PIPE)
            stdout, stderr = heif_2_jpg_process.communicate()
            if heif_2_jpg_process.returncode != 0:
                logger.error('stdout {0} stderr {1}', stdout, stderr)
                raise HEIF2JpgError(
                    'heif2jpg command returned error level {0} for command "{1}"'.format(
                        heif_2_jpg_process.returncode, ' '.join(
                            command +
                            [self.context.request.url]
                        )
                    )
                )
            with open(result_file.name, 'r') as f:
                buffer = f.read()
            mime = self.get_mimetype(buffer)
            self.extension = EXTENSION.get(mime, '.jpg')
            return buffer
        finally:
            os.unlink(heif_file.name)
            os.unlink(result_file.name)

    def set_image_data(self, data):
        self.image = np.frombuffer(data, dtype=self.image.dtype).reshape(self.image.shape)

    def image_data_as_rgb(self, update_image=True):
        if self.image_channels == 4:
            mode = 'BGRA'
        elif self.image_channels == 3:
            mode = 'BGR'
        else:
            mode = 'BGR'
            rgb_copy = np.zeros((self.size[1], self.size[0], 3), self.image.dtype)
            cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR, rgb_copy)
            self.image = rgb_copy
        return mode, self.image.tostring()

    def draw_rectangle(self, x, y, width, height):
        cv2.rectangle(self.image, (int(x), int(y)), (int(x + width), int(y + height)), (255, 255, 255))

    def convert_to_grayscale(self, update_image=True, with_alpha=True):
        image = None
        if self.image_channels >= 3 and with_alpha:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)
        elif self.image_channels >= 3:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        elif self.image_channels == 1:
            # Already grayscale
            image = self.image
        if update_image:
            self.image = image
        elif self.image_depth == np.uint16:
            image = np.array(image, dtype='uint8')
        return image

    def paste(self, other_engine, pos, merge=True):
        if merge and not FILTERS_AVAILABLE:
            raise RuntimeError(
                'You need filters enabled to use paste with merge. Please reinstall ' +
                'thumbor with proper compilation of its filters.')

        self.enable_alpha()
        other_engine.enable_alpha()

        sz = self.size
        other_size = other_engine.size

        mode, data = self.image_data_as_rgb()
        other_mode, other_data = other_engine.image_data_as_rgb()

        imgdata = _composite.apply(
            mode, data, sz[0], sz[1],
            other_data, other_size[0], other_size[1], pos[0], pos[1], merge)

        self.set_image_data(imgdata)

    def enable_alpha(self):
        if self.image_channels < 4:
            with_alpha = np.zeros((self.size[1], self.size[0], 4), self.image.dtype)
            if self.image_channels == 3:
                cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA, with_alpha)
            else:
                cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGRA, with_alpha)
            self.image = with_alpha
