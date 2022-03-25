# Bimetallic Materials for Catalysis: Database Analysis

## Project Overview 

One of the goals of my research group is to use high-throughput computing to find promising material candidates for catalytic reactions that occur in alternative energy resources. Catalytic activity can be predicted by calculating the binding energy of key intermediates of chemical reactions. Combining two transition metals in one material is a good way to create new material candidates as an alloy can have drastically different properties than either of the metals individually. Predicting the catalytic activity of random alloys can be difficult since various binding sites occur on the surface of materials. 

Our research group created a database of binding energies for bimetallic materials where the ensemble, or composition of atom types, is varied. Binding energies were all calculated by students via density functional theory.  Students created backend scripts to accumulate data and frontend scripts to display our results on our interactive webpage:

http://fri.oden.utexas.edu/~fri/fridb/server.py

## Repository Details

This repo contains the data accumulated from this project in the bimetallic_data.json.  It also includes a jupyter notebook that displays and analysis the data accumulated and applies machine learning and statistical methods to predict binding energies of new bimetallic materials

