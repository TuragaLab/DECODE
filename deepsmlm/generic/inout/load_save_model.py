import torch


class LoadSaveModel:
    def __init__(self, model_instance, output_file, input_file=None):
        self.warmstart_file = input_file
        self.output_file = output_file
        self.output_file_suffix = 0
        self.model = model_instance

    def load_init(self, cuda=torch.cuda.is_available()):
        model = self.model
        print('Model instantiated.')
        if self.warmstart_file is None:
            # model.weight_init()
            print('Model initialised randomly as specified in the constructor.')
        else:

            if cuda:
                model.load_state_dict(torch.load(self.warmstart_file))
            else:
                model.load_state_dict(torch.load(self.warmstart_file, map_location='cpu'))

            print('Loaded pretrained model: {}'.format(self.warmstart_file))
        model.eval()
        return model

    def save(self, model, trigger_new_name=False):
        if trigger_new_name:
            self.output_file_suffix += 1

        # a bit hacky
        fname = self.output_file[:-3] + '_' + str(self.output_file_suffix) + '.pt'
        torch.save(model.state_dict(), fname)
        print('Saved model to file: {}'.format(fname))

