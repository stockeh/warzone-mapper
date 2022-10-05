import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import cv2
import sys
import os


class Stitch():

    def __init__(self, capfilename, bgfilename, capture_rate=1, max_frames=-1,
                 mini_param=1080, method='sift') -> None:
        # stock mini_params (21, 191), (29, 199)
        # huskerrs mini_params (27, 286), (41, 300)
        if mini_param == 1080:
            mini_params = ((21, 191), (29, 199))
        elif mini_param == 1440:
            mini_params = ((27, 286), (41, 300))
        else:
            raise AssertionError('Invalid mini_param, must be 1080 or 1440.')

        self.capfilename = capfilename
        self.bgfilename = bgfilename

        # dimensions for verdansk map
        self.bgimage = cv2.imread(bgfilename)
        if 'verdansk' in bgfilename:
            self.bgimage = self.bgimage[240:1720, 280:1660]

        self.images = self._video_to_images(
            mini_params, capture_rate, max_frames)

        descriptors = [
            cv2.xfeatures2d.SIFT_create,
            cv2.BRISK_create,
            cv2.ORB_create,
            cv2.KAZE_create,
            cv2.AKAZE_create
        ]
        names = [str(o.__name__).split('_')[0].lower() for o in descriptors]
        try:
            self.descriptor = descriptors[names.index(str(method).lower())]()
        except:
            raise NotImplementedError(
                f'init: {method=} is not yet implemented for descriptor.')
        if method == 'sift' or method == 'surf':
            self.matcher = cv2.BFMatcher_create(cv2.NORM_L2)
        else:
            self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        self.path_src = None
        self.path_center = None

    def __repr__(self):
        str = f'Stitch({self.capfilename}, {self.bgfilename})'
        return str

    def _video_to_images(self, mini_params, capture_rate=1, max_frames=-1):
        start_t = time.time()
        video = cv2.VideoCapture(self.capfilename)
        images = []
        fps = video.get(cv2.CAP_PROP_FPS)
        print(f'{fps=:.3f}, capturing frame every {capture_rate} seconds.')

        frame = 0
        while video.isOpened():
            success, img = video.read()
            if not success:
                break
            if (frame % (np.floor(fps) * capture_rate) == 0) or capture_rate == -1:
                images.append(img)
            frame += 1
            if max_frames != -1 and len(images) == max_frames:
                print('Max number of frames captured.')
                break

        video.release()
        cv2.destroyAllWindows()

        images = np.array(images)
        if mini_params is not None:
            # different capture resolutions will have different minimap locations
            assert len(mini_params) == 2 and len(
                mini_params[0]) == 2 and len(mini_params[1]) == 2, \
                'Invalid mini_params. Must be ((min,max),(min,max))'
            images = images[:, mini_params[0][0]:mini_params[0][1],
                            mini_params[1][0]:mini_params[1][1]]
        print(f'Finished in {time.time() - start_t:.3f}')

        return images

    def _detect_and_describe(self, image, descriptor, verbose):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if verbose:
            plt.imshow(gray, cmap='gray')
            plt.axis('off')
            plt.show()
        kp, des = descriptor.detectAndCompute(gray, None)
        return {'kp': kp, 'des': des}

    def _combine(self, imga, imgb, verbose=False):
        iset1 = self._detect_and_describe(imga, self.descriptor, verbose)
        iset2 = self._detect_and_describe(imgb, self.descriptor, verbose)

        matches = self.matcher.knnMatch(iset1['des'], iset2['des'], k=2)

        good_matches = []
        for m, n in matches:
            # TODO: is 0.7 enough for the Caldera map?
            # Can’t find enough keypoints - only have 0.
            if m.distance < 0.7*n.distance:
                good_matches.append([m])

        if verbose:
            img3 = cv2.drawMatchesKnn(imga, iset1['kp'], imgb, iset2['kp'], good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
            ax.axis('off')
            plt.show()
            # fig.savefig('../media/knnvectors.png',
            #             dpi=300, bbox_inches='tight')

        if len(good_matches) > 3:
            kp1 = iset1['kp']
            kp2 = iset2['kp']

            src_pts = np.float32(
                [kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Calculate Homography
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        else:
            raise AssertionError(
                f'Can’t find enough keypoints - only have {len(good_matches)}.')

        # Warp source image to destination based on homography
        warp_src = cv2.warpPerspective(
            imga, H, (imgb.shape[1], imgb.shape[0]))

        return warp_src

    def _warp(self, kills):
        # TODO: kills is not implemented yet.
        warps = []
        for imga in self.images:
            try:
                warps.append(self._combine(imga, self.bgimage, verbose=False))
            except Exception as e:
                continue  # print(e)

        return warps

    def _merge(self, warps, threshold):
        src = None
        centers = []
        valid_inds = []

        for i, warp in enumerate(warps):
            nonzero = np.column_stack(np.nonzero(warp))
            ymin, ymax = np.min(nonzero[:, 0]), np.max(nonzero[:, 0])
            xmin, xmax = np.min(nonzero[:, 1]), np.max(nonzero[:, 1])
            if ymax - ymin < threshold and xmax - xmin < threshold:
                row = round(np.mean([ymin, ymax]))
                col = round(np.mean([xmin, xmax]))
                centers.append([col, row])
                if src is None:
                    src = warp
                else:
                    src = np.where(src != 0, src, warp)
                valid_inds.append(i)
            else:
                continue  # print(f'Invalid warp: {i=}, {percent_nonzero:.3f}')

        return src, np.array(centers)

    def compute(self, threshold=250, kills=False):  # verdansk 150
        # TODO: need a better way to evaluate threshold
        warps = self._warp(kills)
        print('Finished warping images.')
        self.path_src, self.path_center = self._merge(warps, threshold)
        while self.path_src is None:
            threshold += 50
            self.path_src, self.path_center = self._merge(warps, threshold)
        print('Finished merging images.')

    def draw_single(self, index):
        assert index < len(
            self.images), f'Invalid index specified. Must be within [0,{len(self.images) - 1}]'
        # self._combine(self.images[index], self.bgimage, verbose=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(cv2.cvtColor(self.images[index], cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.show()

    def draw(self, cmap=plt.cm.cool, filename=None):
        # Blend the warped image and the destination image
        alpha = 0.45
        beta = (1.0 - alpha)
        dst_warp_blended = cv2.addWeighted(
            self.bgimage, alpha, self.path_src, beta, 0.0)

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(cv2.cvtColor(dst_warp_blended, cv2.COLOR_BGR2RGB))
        # solid color
        # ax.plot(self.path_center[:, 0], self.path_center[:, 1], 'x-', color='red', linewidth=1.,
        #         markersize=4, mec='white')
        # multiple colors
        xy = np.vstack([self.path_center[:, 0], self.path_center[:, 1]]).T
        color = [cmap(i*2) for i in np.linspace(0, 1, xy.size)]
        for i, (start, stop) in enumerate(zip(xy[:-1], xy[1:])):
            x, y = zip(start, stop)
            ax.plot(x, y, 'x-', color=color[i],
                    linewidth=1., markersize=4, mec='white')
        ax.axis('off')
        fig.tight_layout()
        if filename:
            fig.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()


def main(args):
    mapper = Stitch(args.input, args.background,
                    capture_rate=5, max_frames=100,
                    mini_params=((27, 286), (41, 300)), method='sift')

    mapper.compute()

    # TODO: temp verbose debug of background image
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(cv2.cvtColor(mapper.bgimage, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    plt.show()

    mapper.draw(filename=args.output)


if __name__ == '__main__':
    """
    Usage: python -u -m mapper.wzmapper -i /Users/stock/Downloads/MWF006.mp4 -b /Users/stock/Downloads/compass_map_mp_don4.jpg
    """
    try:
        parser = argparse.ArgumentParser(description='configuration')
        # TODO: default input does not exist for anyone else
        parser.add_argument('-i', '--input', metavar='path', type=str, default='/Users/stock/Downloads/MWF006.mp4',
                            help='the path to input video file')
        parser.add_argument('-b', '--background', metavar='path', type=str,
                            default=os.path.join(os.path.dirname(
                                os.path.realpath(__file__)), '../docs/verdansk.jpg'),
                            help='the path to background image')
        parser.add_argument('-o', '--output', metavar='path', type=str,
                            help='the path to output image file')
        args = parser.parse_args()
        print(args.background)
        main(args)
    except Exception as e:
        print(e)
        sys.exit()
