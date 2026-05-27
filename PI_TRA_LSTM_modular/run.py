# -*- coding: utf-8 -*-
try:
    from PI_TRA_LSTM_modular.main import main
except ModuleNotFoundError:
    from PI_TRA_LSTM_modular_标准版.main import main

if __name__ == "__main__":
    main()
