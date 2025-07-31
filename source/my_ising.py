import torch
import numpy as np


def energy(sample, ham, lattice):
    spins = sample[:, :, :, :, 1] - sample[:, :, :, :, 0] 
    
    term_x = spins[:, :, 1:, :] * spins[:, :, :-1, :]
    term_x = term_x.sum(dim=(1, 2, 3))
    
    term_y = spins[:, :, :, 1:] * spins[:, :, :, :-1]
    term_y = term_y.sum(dim=(1, 2, 3))
    
    output = term_x + term_y
    
    term_x_periodic = spins[:, :, 0, :] * spins[:, :, -1, :]
    term_x_periodic = term_x_periodic.sum(dim=(1, 2))
    output += term_x_periodic
    
    term_y_periodic = spins[:, :, :, 0] * spins[:, :, :, -1]
    term_y_periodic = term_y_periodic.sum(dim=(1, 2))
    output += term_y_periodic
    
    if ham == 'fm':
        output *= -1

    return output


def magnetization(sample):
    spins = sample[:, :, :, :, 1] - sample[:, :, :, :, 0]
    mag = spins.sum(dim=(1, 2, 3))
    return mag


def neighbors_sum(sample):

    spins = sample[:, :, :, :, 1] - sample[:, :, :, :, 0] 
    
    term = torch.cat((spins[:, :, -1:, :], spins[:, :, :-1, :]), dim=2)
    output = term
    
  
    term = torch.cat((spins[:, :, 1:, :], spins[:, :, :1, :]), dim=2)
    output += term

    term = torch.cat((spins[:, :, :, -1:], spins[:, :, :, :-1]), dim=3)
    output += term
    
    term = torch.cat((spins[:, :, :, 1:], spins[:, :, :, :1]), dim=3)
    output += term
    
    return output