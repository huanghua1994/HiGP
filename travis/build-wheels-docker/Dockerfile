FROM quay.io/pypa/manylinux2014_x86_64

RUN yum update
RUN yum install lsof vim -y

COPY ./bashrc.txt /root/.bashrc
COPY ./vimrc.txt  /root/.vimrc
COPY ./install-conda-envs.sh /tmp/install-conda-envs.sh
RUN chmod +x /tmp/install-conda-envs.sh
RUN /tmp/install-conda-envs.sh
