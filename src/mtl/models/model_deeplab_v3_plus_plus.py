import torch
import torch.nn.functional as F

from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p


class ModelDeepLabV3PlusPlus(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc
        ch_out = sum(outputs_desc.values())

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=True,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.aspp_seg = ASPP(ch_out_encoder_bottleneck, 256)

        self.decoder_seg = DecoderDeeplabV3p(256, ch_out_encoder_4x, self.outputs_desc[list(self.outputs_desc.keys())[0]])

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        # print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))
        
        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]
        features_tasks_seg = self.aspp_seg(features_lowest)

        predictions_4x_seg, _ = self.decoder_seg(features_tasks_seg, features[4])

        # channel size of 19
        predictions_1x_seg = F.interpolate(predictions_4x_seg, size=input_resolution, mode='bilinear', align_corners=False)

        out = {}
        offset = 0

        # for task, num_ch in self.outputs_desc.items():
        #     out[task] = predictions_1x[:, offset:offset+num_ch, :, :]
        #     offset += num_ch

        out[list(self.outputs_desc.keys())[0]]=predictions_1x_seg

        return out
