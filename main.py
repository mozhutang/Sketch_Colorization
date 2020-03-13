import torch
import GetArgs
import Prepare
import Train
import Predict

# Preset arguments
datapath = './data'
style = 'style1.png'
test = 'test1.png'

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get arguments
parser = GetArgs.GetArgs()
args = parser.parse_args()

# Caculate color histogram
Prepare.CaculateColorHisto(datapath)

# Train
Train.train(args, device, datapath)

# Predict
Predict.Predict(datapath, style, test)

