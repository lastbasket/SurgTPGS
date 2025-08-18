import torch
from torch import nn


class LanguageDeformation(nn.Module):
    def __init__(self, args) :
        super(LanguageDeformation, self).__init__()
        self.args = args
        
        self.conv_alpha = nn.Conv1d(in_channels=args.lang_in_ch, out_channels=args.lang_out_ch_beta, kernel_size=1)
        self.conv_beta = nn.Conv1d(in_channels=args.lang_in_ch, out_channels=args.lang_out_ch_beta, kernel_size=1)
        self.slope = args.slope
        
    
    def custom_sigmoid(self, x, slope=2.5):
        """
        Custom sigmoid function with a specified slope.

        Args:
            x (torch.Tensor): Input tensor.
            slope (float): Slope parameter for the sigmoid function.

        Returns:
            torch.Tensor: Output tensor after applying the custom sigmoid function.
        """
        return 1 / (1 + torch.exp(-slope * x))
    
    
    def forward(self, lang_fea, deform_coefs, time):
        t_tensor = torch.full((1,lang_fea.size(0)), time, dtype=lang_fea.dtype, device=lang_fea.device)
        combined_feature = torch.cat([lang_fea.permute(1,0), t_tensor], dim=0)  # [N, 4]
        sigmoid_deform = self.custom_sigmoid(deform_coefs, self.slope)
        
        alpha = self.conv_alpha(combined_feature).squeeze(-1).permute(1,0)  # [N, 3]
        beta = self.conv_beta(combined_feature).squeeze(-1).permute(1,0)    # [N, 3]
        
        lang_fea_deform =  lang_fea + beta * sigmoid_deform  #* sigmoid_deform *alpha 
        # lang_fea_deform =  lang_fea* sigmoid_deform *alpha 
        
        return lang_fea_deform