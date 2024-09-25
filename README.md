# Setup
1. Clone Open-GroundingDino
```
git clone https://github.com/longzw1997/Open-GroundingDino.git && cd Open-GroundingDino/
```
2. Copy open_groundingdino_handover file to Open-GroundingDino
3. Install the required dependencies.
```
pip install -r requirements.txt 
pip install -r requirements.txt 
cd models/GroundingDINO/ops
python setup.py build install
python test.py
cd ../../..
```
