import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from train_transformer import ActionToVerbTransformer, CalvinVerbDataset
from utils import load_calvin_to_dataframe

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Could not find model weights at {args.model_path}")
        
    print(f"Loading weights from {args.model_path}...")
    state_dict = torch.load(args.model_path, map_location=device)
    
    # Dynamically extract num_verbs from the last classifier layer's bias
    num_verbs = state_dict['classifier.4.bias'].shape[0]
    print(f"Detected a model trained to predict {num_verbs} verbs.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # Load the test dataset
    print(f"Loading test dataset from {args.data_dir}...")
    df = load_calvin_to_dataframe(args.data_dir)
    dataset = CalvinVerbDataset(df, args.data_dir, transform=transform, max_seq_len=args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize and load the model
    model = ActionToVerbTransformer(num_verbs=num_verbs, d_model=64, max_seq_length=args.max_seq_len + 3)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("\nStarting Evaluation...\n")
    correct = 0
    total = 0
    
    # Evaluation Loop
    with torch.no_grad():
        for batch_idx, (first, last, actions, labels) in enumerate(dataloader):
            first, last = first.to(device), last.to(device)
            actions, labels = actions.to(device), labels.to(device)
            
            logits = model(first, last, actions)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Print batch results
            for p, l in zip(preds, labels):
                pred_verb = dataset.id_to_verb[p.item()]
                true_verb = dataset.id_to_verb[l.item()]
                status = "✅" if p.item() == l.item() else "❌"
                print(f"{status} Target: {true_verb:<15} | Predicted: {pred_verb}")

    # Report
    accuracy = 100 * correct / total if total > 0 else 0
    print("\n" + "="*50)
    print(f"TESTING COMPLETE")
    print(f"Total Examples Evaluated: {total}")
    print(f"Overall Accuracy:         {accuracy:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to CALVIN dataset directory for testing")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved .pth model weights")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    main(args)