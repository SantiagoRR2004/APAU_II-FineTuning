# Fine-Tuning for [Named Entity Recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition)

## Overview

This is a project for a subject called "Aprendizaje Automático II" (Machine Learning II) that accounted for 10% of the final grade. It was made by a team of 4 people. The submission took place on Saturday, the 31st of May of 2025, and earned a grade of 10 out of 10 points.

## Project Summary

In this project, we implemented a Named Entity Recognition (NER) system using machine learning and natural language processing (NLP) techniques. The main goal was to train and evaluate models to identify and classify entities in text. First, we had to propagate the labels in [`ner-es.trainOld.csv`](/data/ner-es.trainOld.csv), then we fine-tuned [`crf.es.model`](/crf.es.model) and finally we used [`ner-es.validOld.csv`](/data/ner-es.validOld.csv) to evaluate the model.
