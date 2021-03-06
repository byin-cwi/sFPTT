import torch
from tqdm import tqdm

from utils.utils import get_lr

global alpha
global beta
global rho


alpha = 0.1
beta = 0.5
rho = .0
is_debias = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_stats_named_params( model ):
    named_params = {}
    for name, param in model.named_parameters():
        sm, lm, dm = param.detach().clone(), 0.0*param.detach().clone(), 0.0*param.detach().clone()
        named_params[name] = (param, sm, lm, dm)
    return named_params

# def pre_pre_optimizer_updates( named_params):
#     if not is_debias: return
#     for name in named_params:
#         param, sm, lm, dm = named_params[name]
#         param_data = param.data.clone()
#         param.data.copy_( sm.data )
#         sm.data.copy_( param_data )
#         del param_data

# def pre_optimizer_updates( named_params):
#     if not is_debias: return
#     for name in named_params:
#         param, sm, lm, dm = named_params[name]
#         lm.data.copy_( param.grad.detach() )
#         param_data = param.data.clone()
#         param.data.copy_( sm.data )
#         sm.data.copy_( param_data )
#         del param_data

def post_optimizer_updates( named_params, epoch ):
    alpha = 0.5# 0.1
    beta = 0.5
    rho = .0
    is_debias = False
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        if is_debias:
            beta = (1. / (1. + epoch))
            sm.data.mul_( (1.0-beta) )
            sm.data.add_( beta * param )

            rho = (1. / (1. + epoch))
            dm.data.mul_( (1.-rho) )
            dm.data.add_( rho * lm )
        else:
            lm.data.add_( -alpha * (param - sm) )
            sm.data.mul_( (1.0-beta) )
            sm.data.add_( beta * param - (beta/alpha) * lm )

def get_regularizer_named_params( named_params, _lambda=1.0 ):
    regularization = torch.zeros( [], device=device )
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        regularization += (rho-1.) * torch.sum( param * lm )
        if is_debias:
            regularization += (1.-rho) * torch.sum( param * dm )
        else:
            r_p = _lambda * 0.5 * alpha * torch.sum( torch.square(param - sm) )
            regularization += r_p

    return regularization 

def reset_named_params(named_params):
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        param.data.copy_(sm.data)
        
        
def fit_one_epoch(model_train, model, yolo_loss, loss_history, 
                  optimizer, epoch, epoch_step, epoch_step_val, 
                  gen, gen_val, Epoch, cuda, named_params):
    loss        = 0
    reg_loss    = 0
    val_loss    = 0
    fr_loss = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
    
            for p in range(model_train.p):
                #----------------------#
                #   ????????????
                #----------------------#
                optimizer.zero_grad()
                if p ==0:
                    hidden = model_train.init_hidden(images.shape[0])
                else:
                    hidden = tuple(v.detach() for v in hidden)
                #----------------------#
                #   ????????????
                #----------------------#
            
                outputs,hidden,fr_    = model_train.network(images,hidden)

                fr_loss += fr_
                loss_value_all  = 0
                num_pos_all     = 0
                #----------------------#
                #   ????????????
                #----------------------#
                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                loss_value = loss_value_all / num_pos_all
                
                regularizer = get_regularizer_named_params( named_params,  _lambda=1.0 ) 

                loss_reg = loss_value + regularizer
                #----------------------#
                #   ????????????
                #----------------------#
                loss_reg.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                
                post_optimizer_updates( named_params,epoch )

                loss += loss_value.item()
                reg_loss += regularizer.item()
            
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1)/model_train.p, 
                                'reg'   : reg_loss / (iteration + 1), 
                                'fr'   : fr_loss / (iteration + 1)/model_train.p, 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)
            
    reset_named_params(named_params)
    
    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                #----------------------#
                #   ????????????
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   ????????????
                #----------------------#
                outputs         = model_train(images)

                loss_value_all  = 0
                num_pos_all     = 0
                #----------------------#
                #   ????????????
                #----------------------#
                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                loss_value  = loss_value_all / num_pos_all

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    
    loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
