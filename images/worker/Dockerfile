FROM  --platform=linux/amd64 centos:8

RUN cd /etc/yum.repos.d/
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*
#Update
RUN yum update -y

# Dependencies
RUN yum -y install wget

#Install Python 3
RUN dnf install -y python3
RUN dnf install -y  python2 python2-chardet
RUN dnf install -y policycoreutils-python-utils




# Install yum repository
#RUN cd /etc/yum.repos.d && wget https://research.cs.wisc.edu/htcondor/yum/repo.d/htcondor-stable-rhel8.repo

# Import signing key
#RUN wget http://research.cs.wisc.edu/htcondor/yum/RPM-GPG-KEY-HTCondor && rpm --import RPM-GPG-KEY-HTCondor

RUN  dnf --enablerepo=powertools install boost-python3 -y
# Install HTCondor
#RUN yum -y install condor
RUN yum install -y https://research.cs.wisc.edu/htcondor/repo/9.x/htcondor-release-current.el8.noarch.rpm
RUN yum install -y condor
# Install Nano
RUN yum -y install nano


COPY condor_config.local /etc/condor/
COPY worker /etc/condor/config.d/
COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]

