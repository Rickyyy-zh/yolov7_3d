import os
import os.path as osp
import argparse
from show_2d3d_box import *
from tqdm import tqdm
from utils.datasets import letterbox, random_perspective_3D

color_list = {'car': (0, 0, 255),
              'truck': (0, 255, 255),
              'van': (255, 0, 255),
              'bus': (255, 255, 0),
              'cyclist': (0, 128, 128),
              'motorcyclist': (128, 0, 128),
              'tricyclist': (128, 128, 0),
              'pedestrian': (0, 128, 255),
              'barrow': (255, 0, 128)}
type_list = {'car': 0,
              'truck': 1,
              'van': 2,
              'bus': 3,
              'cyclist': 4,
              'motorcyclist': 5,
              'tricyclist': 6,
              'pedestrian': 7,
              'barrow': 8}

def vis_label_in_image_txt(path, save_path, thresh):
    label_path = osp.join(path, "label/")
    img_path = osp.join(path, "image/")
    calib_path = osp.join(path, "calib/")
    denorm_path = osp.join(path, "denorm/")
    ext_path = osp.join(path, "extrinsics/")


    set_path = osp.join(path, "train.txt")
    with open(set_path, 'r') as f:
        file_names = f.read().strip().splitlines()
        for file in tqdm(file_names):
            p_lb = osp.join(label_path, file + ".txt")
            P_im = osp.join(img_path, file + ".jpg")
            P_ca = osp.join(calib_path, file + ".txt")
            P_de = osp.join(denorm_path, file + ".yaml")
            P_ex = osp.join(ext_path, file + ".yaml")

            result = load_label_data(p_lb)
            world2camera = read_kitti_ext(P_ex).reshape((4, 4))
            camera2world = np.linalg.inv(world2camera).reshape(4, 4)
            p2 = read_kitti_cal(P_ca)
            img = cv2.imread(P_im)
            h, w, c = img.shape
            # Resize 
            # img  = cv2.resize(img, (960, 640))

            # ratio_x = 960 / w
            # ratio_y = 640 / h 


            # padding 
            # img, ratio, pad = letterbox(img, (1080, 1920), auto=False, scaleup=True)
            # ratio_x = ratio[0]
            # ratio_y = ratio[1]

            # p2[0,0] = p2[0,0] * ratio_x 
            # p2[0,2] = p2[0,2] * ratio_x + pad[0]    #中心偏移
            # p2[1,1] = p2[1,1] * ratio_y 
            # p2[1,2] = p2[1,2] * ratio_y + pad[1]    #中心偏移
            label = []
            for i in result:
                if i[-1] < thresh:
                    continue
                if i[0] not in type_list.keys():
                    continue
                if i[8] <= 0.05 and i[9] <= 0.05 and i[10] <= 0.05: #invalid annotation
                    continue
                i[0] = type_list[i[0]]
                label.append(i)
            label = np.array(label)
            img, label, p2 = random_perspective_3D(img, label, p2, degrees=30,
                                                 translate=0.4,
                                                 scale=0.5,
                                                 shear=20,
                                                 perspective=0)
            
            save_flg = False
            color_dic = {k:i for i,k in type_list.items()}
            for result_index in range(label.shape[0]):
                t = label[result_index]
                color_type = color_list[color_dic[t[0]]]
                cam_bottom_center = [t[11], t[12], t[13]]  # bottom center in Camera coordinate

                center_2d = [int((t[4] + t[6]) / 2), int((t[5] + t[7]) / 2)]
                center_3d_ground = p2[:3, :3] * np.matrix(cam_bottom_center).T
                center_3d_ground = center_3d_ground / center_3d_ground[2]
                center_3d_ground_pixel = [int(center_3d_ground[0]), int(center_3d_ground[1])]

                center_3d_box = p2[:3, :3] * (np.matrix(cam_bottom_center) - np.array((0 , t[8] * 0.5, 0))).T

                center_3d_box = center_3d_box / center_3d_box[2]
                center_3d_box_pixel = [int(center_3d_box[0]), int(center_3d_box[1])]

                if center_3d_box_pixel[1] > center_3d_ground_pixel[1]:
                    save_flg = True
                    print(file)
                # else:
                #     break
                cv2.rectangle(img, (int(t[4]), int(t[5])), (int(t[6]), int(t[7])),
                            (255, 255, 255), 1)

                bottom_center_in_world = camera2world * np.matrix(cam_bottom_center + [1.0]).T
                verts3d = project_3d_world(p2, bottom_center_in_world, t[9], t[8], t[10], t[14], camera2world)
                
                if verts3d is None:
                    continue
                verts3d = verts3d.astype(np.int32)

                # draw projection
                cv2.line(img, tuple(verts3d[2]), tuple(verts3d[1]), color_type, 2)
                cv2.line(img, tuple(verts3d[1]), tuple(verts3d[0]), color_type, 2)
                cv2.line(img, tuple(verts3d[0]), tuple(verts3d[3]), color_type, 2)
                cv2.line(img, tuple(verts3d[2]), tuple(verts3d[3]), color_type, 2)
                cv2.line(img, tuple(verts3d[7]), tuple(verts3d[4]), color_type, 2)
                cv2.line(img, tuple(verts3d[4]), tuple(verts3d[5]), color_type, 2)
                cv2.line(img, tuple(verts3d[5]), tuple(verts3d[6]), color_type, 2)
                cv2.line(img, tuple(verts3d[6]), tuple(verts3d[7]), color_type, 2)
                cv2.line(img, tuple(verts3d[7]), tuple(verts3d[3]), color_type, 2)
                cv2.line(img, tuple(verts3d[1]), tuple(verts3d[5]), color_type, 2)
                cv2.line(img, tuple(verts3d[0]), tuple(verts3d[4]), color_type, 2)
                cv2.line(img, tuple(verts3d[2]), tuple(verts3d[6]), color_type, 2)

                # if t.h * t.Y > 0:
                #     print(cam_bottom_center)
                #     print(t.h)
                #     cv2.rectangle(img, (t.x1, t.y1), (t.x2, t.y2),(255, 255, 255), 4)
                #     center_3d_box = p2[:3, :3] * (np.matrix(cam_bottom_center) - np.array((0 , t.h * 0.5, 0))).T
                #     print((np.matrix(cam_bottom_center) - np.array((0 , t.h * 0.5, 0))))
                # else:
                #     center_3d_box = p2[:3, :3] * (np.matrix(cam_bottom_center) - np.array((0 , t.h * 0.5, 0))).T
                #     print(cam_bottom_center)
                #     print(t.h)
                #     print((np.matrix(cam_bottom_center) - np.array((0 , t.h * 0.5, 0))))


# 标注3d中心y值
                cv2.circle(img, (center_2d[0], center_2d[1]), 2, (0,0,0), 2)
                cv2.circle(img, (center_3d_ground_pixel[0], center_3d_ground_pixel[1]), 2, (255,0,0), 2)
                cv2.circle(img, (center_3d_box_pixel[0], center_3d_box_pixel[1]), 2, (0,255,0), 2)
                box_3d_loc_str = str(round(cam_bottom_center[0],3)) + ", " + str(round(cam_bottom_center[1] + t[8] * 0.5, 3)) + ", " + str(round(cam_bottom_center[2],3))
                cv2.putText(img, box_3d_loc_str, (center_3d_box_pixel[0], center_3d_box_pixel[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)

            if not save_flg:
                cv2.imwrite('%s/%s.jpg' % (save_path, file), img)


def add_arguments(parser):
    parser.add_argument(
        "--path",
        type=str,
        default="visualize/rope3d-dataset-tools/show_tools/data_demo/",
    )
    parser.add_argument("--output-file", type=str, default="visualize/rope3d_mini_res/")
    parser.add_argument("--thres", type=float, default=0.0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    if not osp.exists(args.output_file):
        os.mkdir(args.output_file)

    # vis_label_in_image(args.path, args.output_file)
    vis_label_in_image_txt(args.path, args.output_file, args.thres)

