FROM archlinux/base
RUN pacman -Syu --noconfirm
RUN pacman -S python3 python-pip python-setuptools opencv hdf5 --noconfirm
RUN pacman -S base-devel --noconfirm
COPY ./FreeImage /tmp/FreeImage
RUN cd /tmp/FreeImage && make -j2 && make install 
RUN pip install scikit-learn mahotas python-telegram-bot
RUN pacman -Rs gcc make automake --noconfirm
RUN pacman -S qt5-base --noconfirm
COPY ./*.py /app/
COPY ./data /app/data
CMD cd /app && python3 ./main.py
