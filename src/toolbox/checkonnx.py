import sys

import cv2
import numpy as np
import onnxruntime

'''
check an onnx model.
Usage: python checkonnx.py <onnx_path> <input_img_path> <output_img_path>
'''
assert len(sys.argv) == 4
print('[ ARGS ] onnx_path:', sys.argv[1])
print('[ ARGS ] input_img_path:', sys.argv[2])
print('[ ARGS ] output_img_path:', sys.argv[3])


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("[ INFO ] input_name: {}".format(self.input_name))
        print("[ INFO ] output_name: {}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        sr = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return sr

    def forward2(self, image_tensor1, image_tensor2):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''

        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        sr = self.onnx_session.run(None, {self.input_name[0]: image_tensor1, self.input_name[1]: image_tensor2})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        # input_feed = self.get_input_feed(self.input_name, [ image_tensor1, image_tensor2 ])
        # sr = self.onnx_session.run(self.output_name, input_feed=input_feed)

        return sr


def main():
    frame0 = cv2.imread(sys.argv[2])
    ins = frame0.astype(np.float32)
    ins = np.transpose(ins, [2, 0, 1])
    ins = ins / 255.
    ins = ins[np.newaxis, :, :, :]
    print('[ INPUT ] image shape is:', ins.shape)

    # load onnx
    sr_model = ONNXModel(sys.argv[1])
    out_final = sr_model.forward(ins)

    # save output 
    img_cons = out_final[0][0]
    img_cons = img_cons * 255.0
    img_cons = np.clip(img_cons, 0, 255).astype(np.uint8)
    img_cons = np.transpose(img_cons, [1, 2, 0])

    cv2.imwrite(sys.argv[3],
                img_cons,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    main()
