## Session 1 - Background & Basics:Machine Learning Intuition

### Links

- [Eye Painting time lapse](https://youtu.be/jC6qegT972c)
- [Playing with layers](http://scs.ryerson.ca/~aharley/vis/conv/flat.html)

#### Discussion

- Humans use 3 channels(RGB), a newspaper use 4 channels(CYMK).
- Whatever we see is etched on the brain.
- Brain has several edge detectors
- Edges/Texture/Patterns/Parts/Objects
- A channel will consist of only same or similar  context/features
- Channel - Example of lower case and upper case alphabets( 26 and 52 channels)
- A kernel/filter/feature extractor is going to extract a feature for us
- To extract 26 channels here, we need 26 kernels

- Four layers: Edges and Gradiens -> Textures and Patterns -> Parts of objects -> Objects

- 3x3 convolved with 3x3 gives 1
- 5x5 convolved with 5x5 gives 1 = 25 parameters
- 5x5 convolved with 3x3 gives 3x3 --> Now 3x3 convolved with 3x3 gives 1
