# Changelog
All notable changes to this project will be documented in this file.

## [0.11.0]
### Added
- New console script entrypoint for training. Write `decode.train` instead of `python -m decode.neuralfitter.train.live_engine`
- EmitterSet feature a new frame selecting method. `emitter.iframe[2:5]` selects emitters from frame 2 to 5 (upper exclusive).

### Changed
- Pythonic indexing for all integer access. This mainly affects EmitterSet methods such 
  as `get_subset`, `.split_in_frames` method.

### Removed


## [0.10.0]
### Added
- EmitterSet now implements "+" operator which concatenates EmitterSets
- EmitterSet now has chunk method to split an EmitterSet into equally sized chunks
- Plotting supports axes swap `axes_order='future'` because the decode convention will change

### Changed
- EmitterSet implements `.save()` method which now supports .hdf5, .pt (pytorch standard) and .csv.
- EmitterSet implements `.load()` method which supports .hdf5 and .pt (pytorch standard).
- Console device argument must take expalining string not integer. I.e. `-i 1` won't work, it must be `-i cuda:1`

### Removed
