import os
import sys
# 服务器运行一般后台运行， nohub python app_run.py &


sys.path.append(os.path.dirname(__file__))


if __name__ == '__main__':
    from app import app
    app.run(host='0.0.0.0', port=9999)
