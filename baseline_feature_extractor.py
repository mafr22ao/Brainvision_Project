def baseline_feature_extractor():
    """
    reads in unprocessed videos and stores the feature maps of the basic ResNet50
    :return: saves the feature maps
    """
    #
    # def get_activations_and_save_image_model(model, video_list, activations_dir):
    #     """This function generates Alexnet features and save them in a specified directory.
    #     Parameters
    #     ----------
    #     model :
    #         pytorch model : alexnet.
    #     video_list : list
    #         the list contains path to all videos.
    #     activations_dir : str
    #         save path for extracted features.
    #     """
    #
    #     resize_normalize = trn.Compose([
    #         trn.Resize((224, 224)),
    #         trn.ToTensor(),
    #         trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #
    #     for video_file in tqdm(video_list):
    #         vid, num_frames = sample_video_from_mp4(video_file)
    #         video_file_name = os.path.split(video_file)[-1].split(".")[0]
    #         activations = {}
    #         for frame, img in enumerate(vid):
    #             input_img = V(resize_normalize(img).unsqueeze(0))
    #             if torch.cuda.is_available():
    #                 input_img = input_img.cuda()
    #             model_output, img_feature = model(input_img)
    #             for layer_name, f in img_feature.items():
    #                 if frame == 0:
    #                     activations[layer_name] = f.data.cpu().numpy().ravel()
    #                 else:
    #                     activations[layer_name] += f.data.cpu().numpy().ravel()
    #         for layer_name, f in img_feature.items():
    #             save_path = os.path.join(activations_dir, video_file_name + "_" + layer_name + ".npy")
    #             avg_layer_activation = activations[layer_name] / float(num_frames)
    #             np.save(save_path, avg_layer_activation)

    print("This has not be implemented yet.")