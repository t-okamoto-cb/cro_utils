"""
公式PPTテーマカラー定義
"""
import matplotlib.pyplot as plt
from cycler import cycler

# 個別カラー定数
DARK_NAVY  = "#2B336C"  # ダークネイビー
LIGHT_BLUE = "#7997F9"  # ライトブルー
BLUE       = "#4D6FF1"  # ブルー
PINK       = "#F7418B"  # ピンク/マゼンタ
SOFT_RED   = "#FB67A0"  # ソフトレッド
GRAY       = "#888888"  # ニュートラルグレー

# デフォルトサイクル順（axes.prop_cycle に対応）
PALETTE = [DARK_NAVY, LIGHT_BLUE, BLUE, PINK, SOFT_RED, GRAY]


def apply_plot_theme():
    """matplotlib の axes.prop_cycle を公式PPTテーマカラーに設定する。"""
    plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)
