import torch

###---Enable Dropout---###
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
###---Save model---###
def save_model_(model,model_name,dataset,pre_len):
    torch.save({'model_state_dict': model.state_dict(),
        }, f"check_points/{model_name}_{dataset}_{str(pre_len)}.pth" )   
    print("Model saved!") 

###---Load model---###        
def load_model_(model,model_name,dataset,pre_len):
    PATH1 = f"check_points/{model_name}_{dataset}_{str(pre_len)}.pth"
    checkpoint = torch.load(PATH1)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded!") 
    return model            