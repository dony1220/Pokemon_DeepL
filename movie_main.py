{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPHPOU8kWeFPrW/V3UTdkRP",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/dony1220/dl/blob/main/movie_main.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# movie_main"
      ],
      "metadata": {
        "id": "P3T4xhQqGDs7"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# !pip install deepctr\n",
        "# !pip install import-ipynb"
      ],
      "metadata": {
        "id": "ZRfZcGAfn1RP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from deepctr.feature_column import SparseFeat,get_feature_names\n",
        "from deepctr.models import FLEN, DeepFM\n",
        "from sklearn.metrics import log_loss, roc_auc_score\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import tensorflow as tf\n",
        "import import_ipynb"
      ],
      "metadata": {
        "id": "bdWvrdmon1Pv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# !pip install tensorflow==2.8.0\n",
        "print(tf.__version__)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2ELpygq8rIk7",
        "outputId": "48014595-88ec-4940-d0c2-12fb0572ee57"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2.8.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "#드라이브 연결"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uwmm6W--ois3",
        "outputId": "b8aabf61-bc01-49a6-b722-023afcf7af3f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%cd \"/content/drive/MyDrive/Colab Notebooks/git_test/dl/dl\""
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RopCDAaGxd6H",
        "outputId": "c0ac09ee-161a-4dad-ca33-cb9ec858c164"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/drive/MyDrive/Colab Notebooks/git_test/dl/dl\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ls"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "b4TY6KAdxx4N",
        "outputId": "0a40c0ed-2ce9-4231-b014-974d0d7f413b"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "FeatureExtractor.py  main.py\t\t     movie_define_model.ipynb  poketest.ipynb\n",
            "ImageProcessing.py   movie_decoding_f.ipynb  movie_main.ipynb\t       README.md\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from movie_define_model import define_model"
      ],
      "metadata": {
        "id": "cjk6_Dkjx0Y2"
      },
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "metadata": {
        "id": "DQcQPMr1ls9I",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e5209502-5eca-46b2-ef50-222dfa668602"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "250/250 - 4s - loss: 0.4868 - binary_crossentropy: 0.4868 - val_loss: 0.4457 - val_binary_crossentropy: 0.4457\n",
            "Epoch 2/10\n",
            "250/250 - 3s - loss: 0.4384 - binary_crossentropy: 0.4383 - val_loss: 0.4446 - val_binary_crossentropy: 0.4444\n",
            "Epoch 3/10\n",
            "250/250 - 3s - loss: 0.4333 - binary_crossentropy: 0.4332 - val_loss: 0.4399 - val_binary_crossentropy: 0.4398\n",
            "Epoch 4/10\n",
            "250/250 - 2s - loss: 0.4279 - binary_crossentropy: 0.4277 - val_loss: 0.4358 - val_binary_crossentropy: 0.4356\n",
            "Epoch 5/10\n",
            "250/250 - 2s - loss: 0.4201 - binary_crossentropy: 0.4200 - val_loss: 0.4278 - val_binary_crossentropy: 0.4276\n",
            "Epoch 6/10\n",
            "250/250 - 2s - loss: 0.4056 - binary_crossentropy: 0.4054 - val_loss: 0.4138 - val_binary_crossentropy: 0.4136\n",
            "Epoch 7/10\n",
            "250/250 - 2s - loss: 0.3862 - binary_crossentropy: 0.3860 - val_loss: 0.4035 - val_binary_crossentropy: 0.4033\n",
            "Epoch 8/10\n",
            "250/250 - 2s - loss: 0.3746 - binary_crossentropy: 0.3743 - val_loss: 0.3924 - val_binary_crossentropy: 0.3921\n",
            "Epoch 9/10\n",
            "250/250 - 2s - loss: 0.3656 - binary_crossentropy: 0.3652 - val_loss: 0.3927 - val_binary_crossentropy: 0.3924\n",
            "Epoch 10/10\n",
            "250/250 - 2s - loss: 0.3572 - binary_crossentropy: 0.3569 - val_loss: 0.3878 - val_binary_crossentropy: 0.3874\n",
            "test LogLoss 0.385\n",
            "test AUC 0.8693\n"
          ]
        }
      ],
      "source": [
        "#define_model이라는 함수에서, df를 들고와 a라는 객체에 저장해주고\n",
        "# 해당 a를 실행했을 때, csv로 바꿔주는!\n",
        "# 해당 저장경로는 따로 설정하기\n",
        "a = define_model()\n",
        "a.to_csv('/content/drive/MyDrive/Colab Notebooks/git_test/dl/dl/final.csv', index=False)"
      ]
    }
  ]
}