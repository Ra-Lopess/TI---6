import { Component } from '@angular/core';
import {HttpClient, HttpHeaders, HttpResponse} from "@angular/common/http";
import axios from 'axios';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'dogs';

  showImageHappy: boolean;
  showImageSad: boolean;
  showImageAngry: boolean;
  showImageRelax: boolean;
  showImageX: boolean;
  isHappy: boolean;
  isSad: boolean;
  isAngry: boolean;
  isRelax: boolean;
  imgSelected: any;
  url: any[];

  constructor(private Http: HttpClient) {
    this.showImageHappy = false;
    this.showImageSad = false;
    this.showImageAngry = false;
    this.showImageRelax = false;
    this.showImageX = false;
    this.isHappy = false;
    this.isSad = false;
    this.isAngry = false;
    this.isRelax = false;
    this.imgSelected = false;
    this.url = [];
  }

  uploadFile($event:any) {
    this.destroyImg();
    for(let i = 0; i < $event.target.files.length; i++){
      this.uploadDocument($event.target.files[i], i);
    }
  }

  uploadDocument(file:any, i:number) {
    let fileReader = new FileReader();

    fileReader.onload = (e) => {
      this.url[i] = fileReader.result;
      console.log(this.url);
    }
    
    this.imgSelected = true;
    return fileReader.readAsDataURL(file);
  }

  destroyImg() {
    this.url = [];
    this.imgSelected = false;
  }

  convertToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = error => reject(error);
    });
  }

  dataURLtoFile(dataurl:any, filename:string) {
    var arr = dataurl.split(','),
        mime = arr[0].match(/:(.*?);/)[1],
        bstr = atob(arr[arr.length - 1]), 
        n = bstr.length, 
        u8arr = new Uint8Array(n);
    while(n--){
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, {type:mime});
}

  async postImg() {
    const base64img = [];
    // this.url.forEach(file => base64img.push(this.convertToBase64(file)));

    const images = []
    for(var i = 0; i < this.url.length; i++){
      var img = {name: `img${i}`, data: await this.convertToBase64(this.dataURLtoFile(this.url[i], "image"))};
      console.log(img)
      images.push(img);
    }

    console.log(images)
    const requestData = JSON.stringify({images});

    axios.post('http://localhost:5000/cnn', requestData, {
      headers: {
        'Content-Type': 'application/json',
      },
    })
    .then(response => {
      console.log(response);
    })
    .catch(error => {
      console.error(error);
    });
  }

  categ(response:string) {
    
      this.isHappy = false;
      this.isSad = false;
      this.isAngry = false;
      this.isRelax = false;

      if(response == 'sad')
        this.isSad = true;
      else if(response == 'relaxed')
        this.isRelax = true;
      else if(response == 'angry')
        this.isAngry = true;
      else if(response == 'happy')
        this.isHappy = true;
      else
        console.log("Não é um Cachorro!")
  }
}
