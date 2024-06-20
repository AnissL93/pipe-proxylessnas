import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# Custom loss function with penalty for segment transitions
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.param = Parameter(torch.Tensor(10,3))

    def forward(self, y_pred):
        # Primary task loss (cross-entropy)
        # primary_loss = F.cross_entropy(y_pred, y_true)

        prob = F.softmax(self.param, dim=-1)
        p_sel = torch.argmax(prob, dim=-1)

        # Penalty for segment transitions based on predictions
        penalty = segment_transition_penalty(p_sel)
        
        # Weight for the penalty term (adjust as needed)
        penalty_weight = 0.1
        p = penalty_weight * penalty,
        print(penalty.item())
        # Combine primary loss and penalty with the desired weight
        total_loss = F.mse_loss(torch.Tensor([p]), torch.Tensor([0.]))
        total_loss.requires_grad_()
        
        return total_loss

# Function to calculate the penalty for segment transitions based on predictions
def segment_transition_penalty(y_pred):
    # Convert predicted sequence to a NumPy array for easier manipulation
    y_pred_np = y_pred
    
    # Find the indices where the transition to a new segment occurs in predictions
    transition_indices = torch.Tensor([1 for i in range(1, len(y_pred_np)) if (y_pred_np[i]-y_pred_np[i-1]) != 0])

    # differences = y_pred_np[1:] - y_pred_np[:-1]
    # print(differences)

    # Find indices where the differences are not equal to 0
    # transition_indices = (differences != 0).nonzero().squeeze() + 1
    print(f"{transition_indices=}")
    transition_indices = transition_indices.float()
    transition_indices.requires_grad_()
    
    # Calculate penalty as the sum of squared differences between consecutive transition indices
    # penalty = sum((transition_indices[i] - transition_indices[i - 1])**2 for i in range(1, len(transition_indices)))
    penalty = (torch.sum(transition_indices) - 2)**2
    
    return penalty

# Example usage:

# Assuming y_true and y_pred are your target and predicted sequences, respectively
y_true = torch.Tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred = torch.Tensor([0, 2, 0, 1, 0, 1, 2, 1, 2])  # Your model's predicted sequence
y_pred = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])  # Your model's predicted sequence

# print(segment_transition_penalty(y_true))
# print(segment_transition_penalty(y_pred))
# print(segment_transition_penalty(p_sel))

# Initialize the custom loss
custom_loss = CustomLoss()
w = torch.rand(10,3)
prob = F.softmax(w, dim=-1)
p_sel = torch.argmax(prob, dim=-1)
print(f"{p_sel=}")

# Calculate and print the total loss
loss = custom_loss(y_true)

optimizer = torch.optim.SGD(custom_loss.parameters(), lr=0.01, momentum=0.9)
optimizer.zero_grad()

loss.backward()

print("gggggggggrad")
print(custom_loss.param.grad)

print(optimizer.step())
print(custom_loss.param)





# import torch
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
# import torch.nn as nn

# lat = torch.rand((3, 10))

# def sp(x, n_proc, n_block):
#     split_point_w = torch.rand((n_proc, n_block))
#     print(f"{split_point_w=}")
#     # split_point_w = torch.Tensor(n_proc, n_block)
#     first_row = split_point_w[:, 0]
#     print(f"{first_row=}")
#     first_proc_prob = F.softmax(first_row, dim=0)
#     print(f"{first_proc_prob=}")
#     first_proc_loc = torch.argmax(first_proc_prob, dim=0).item()
#     print(f"{first_proc_loc=}")

#     sp_prob = F.softmax(split_point_w, dim=-1)
#     print(f"{sp_prob=}")
#     sp_on_procs = torch.argmax(sp_prob, dim=-1).detach()
#     print(f"{sp_on_procs=}")

#     st = 0
#     ed = 0
#     stage0_proc = first_proc_loc
#     sp_on_procs[stage0_proc] = 100
#     stage1_proc = torch.argmin(sp_on_procs).item()
#     sp_on_procs[stage1_proc] = 100
#     stage2_proc = torch.argmin(sp_on_procs).item()
#     print(f"{stage0_proc=}")
#     print(f"{stage1_proc=}")
#     print(f"{stage2_proc=}")


# class PipeSplitPoint(nn.Module):

#     def __init__(self, n_block, n_proc):
#         super(torch.nn.Module, self).__init__()
#         self.split_point_w = torch.Tensor(n_proc, n_block)
#         self.n_proc = n_proc

        
#     def forward(self, x):
#         first_proc_prob = F.softmax(self.split_point_w[0], dim=0)
#         sp_prob = F.softmax(self.split_point_w, dim=-1)

#         first_proc_loc = torch.argmax(first_proc_prob)
#         print(f"{first_proc_loc=}")
#         sp_on_procs = torch.argmax(sp_prob, dim=-1)
#         print(f"{sp_on_procs=}")
#         return sp_on_procs

#     def backward(self, x):
#         pass


# pipe = PipeSplitPoint(10, 3)
# random_tensor = torch.rand((6, 3))
# sp(random_tensor, 3, 6)


# def differentiable_argmax(x, temperature=0.01):
#     logits = x / temperature
#     softmax_values = F.softmax(logits, dim=-1)
#     index = torch.argmax(softmax_values, dim=-1)
#     return index

# # Example data
# input_vector = torch.tensor([2.0, 1.0, 0.1], requires_grad=True)
# target_array = torch.tensor([0.3, 0.6, 0.9])

# # Obtain the differentiable argmax index
# output_index = differentiable_argmax(input_vector)

# # Use the index to access a value from the target array
# selected_value = target_array[output_index]

# # Compute a loss (for example, mean squared error between selected value and a target)
# target_value = torch.tensor(1.0, requires_grad=True)
# loss = F.mse_loss(selected_value, target_value)

# # Backward pass to compute gradients
# loss.backward()

# # Access gradients and perform optimization step (if necessary)
# optimizer = torch.optim.SGD([input_vector], lr=0.1)
# optimizer.step()

# # Print results
# print("Input Vector:", input_vector)
# print("Differentiable Argmax Index:", output_index.item())
# print("Selected Value from Target Array:", selected_value.item())
# print("Loss:", loss.item())
