# StorageSol_ComputerVision
This project will use Convolution Neural Network and other AI techniques to solve my phone's storage issue. It will help me keep those items which I think I care about and rest all would be marked as garbage.

Backgroud: With the advent of data proliferation ( both valuable and garbage) flowing across phones, each one of us face this situation of storage issue and then   started one of the most tedious task of re-claiming our justified storage space back.
WhatsApp image and video forwards and one-click image options doesn’t really leaves us with many choices. We keep storing these images and videos till the point we realise that either our phone slowed -down or we start getting low-storage warnings.

Pre-requisite: 
                    1. You need to first decide all those near and dear one’s of you whose pics you don’t wanted to loose at any cost. 
                    2. Get 2 or 3 good quality, clear pic of those ppl.
                    3. Create a folder structure with each folder containing 1 set of person and folder name as name of that person. Folder content will work as the    input data from Neural Network and folder name will work as the label.
                    

Major Libraries used in the solution:
* PIL
* MTCNN
* Keras_vggface.vggface
* Keras_vggface.utils
* pickles
* imutils
* tensorflow
* keras

Model Building: Train_embedding file is used to build the embedding vector for each face and corresponding label. This model is serialized as a pickle object.

Model Application: Model build in the previous step would be called to deserialized and used to detect faces in the pile of images and take action.

Further details abour this solution avaialble on my blog-post -> https://medium.com/@twarit.nigam/application-of-ai-computer-vision-in-combating-mobile-data-storage-issues-4ff187db10cd?sk=77b319b1cd09b7ad9493f9c3b3773691






